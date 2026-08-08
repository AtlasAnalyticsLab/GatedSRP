"""ABMIL, DSMIL, and TransMIL baselines for frozen WSI patch embeddings.

The models in this file are intentionally isolated from the Gated SRP/XSA/Diff
modules. They let the slide-classification, PANDA, and survival trainers share
the same dataset and optimization interfaces across MIL architectures.

Implementation notes:
  * ABMIL follows the gated attention pooling variant from Ilse et al. because
    that is the commonly used pathology baseline and is provided by the
    official AttentionDeepMIL repository.
  * DSMIL follows the official bag-classifier operation: class-wise critical
    instance selection, query attention against all instances, and a Conv1d bag
    classifier over the class-specific bag representations.
  * OfficialTransMIL preserves the official two-layer TransMIL topology,
    internal width 512, PPEG placement, Nyström settings, and residual
    depthwise V-convolution.  The input projection is adapted from the official
    hard-coded 1024 feature dimension to the frozen encoder dimension used by
    this project, which is necessary for UNI-v2/MedSigLIP/ViT features.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from slide_level.src.nystrom_xsa import moore_penrose_iter_inv


BaselineKind = Literal["abmil", "dsmil", "official_transmil"]


class ABMILAggregator(nn.Module):
    """Gated attention MIL pooling for frozen WSI patch embeddings.

    The official AttentionDeepMIL code maps instance features into a hidden
    space, computes tanh and sigmoid attention branches, multiplies them, and
    normalizes a scalar attention score over instances.  We keep that design
    and only adapt the feature projection to arbitrary frozen encoder dims.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        attention_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.attention_dim = int(attention_dim)

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.attention_v = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attention_dim),
            nn.Tanh(),
        )
        self.attention_u = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attention_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(self.attention_dim, 1)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
        self.last_attention_scores: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"ABMIL expects features with shape (B,N,D), got {tuple(features.shape)}")
        if features.shape[1] <= 0:
            raise ValueError("ABMIL received an empty slide bag.")

        h = self.feature_extractor(features)
        # Gated attention keeps the two official branches separate until the
        # multiplicative fusion; this makes the baseline match the repo variant
        # commonly used in WSI MIL papers.
        attn_logits = self.attention_w(self.attention_v(h) * self.attention_u(h)).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1)
        self.last_attention_scores = attn.detach()
        bag = torch.bmm(attn.unsqueeze(1), h).squeeze(1)
        return self.classifier(bag)


class DSMILAggregator(nn.Module):
    """DSMIL bag classifier adapted to frozen slide-level patch features.

    DSMIL's official BClassifier is class-wise: it selects the highest-scoring
    instance for each class, uses that critical instance as the class query, and
    builds one class-specific bag representation.  The final Conv1d has kernel
    size equal to the hidden feature dimension, matching the official code.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        query_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.query_dim = int(query_dim)

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.instance_classifier = nn.Linear(self.hidden_dim, self.num_classes)
        self.query = nn.Sequential(
            nn.Linear(self.hidden_dim, self.query_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.query_dim, self.query_dim),
            nn.Tanh(),
        )
        self.value = nn.Identity()
        self.bag_classifier = nn.Conv1d(
            self.num_classes,
            self.num_classes,
            kernel_size=self.hidden_dim,
        )
        self.last_attention_scores: torch.Tensor | None = None
        self.last_instance_logits: torch.Tensor | None = None
        # DSMIL is a dual-stream objective, not only a bag classifier.  Keep
        # the current forward's instance logits alive until the trainer forms
        # the class-wise maximum-instance loss.  A private list supports B>1
        # without padding bags or changing the public tensor-only forward API.
        self._instance_logits_for_loss: list[torch.Tensor] | None = None

    def _forward_one(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.shape[0] <= 0:
            raise ValueError("DSMIL received an empty slide bag.")

        h = self.feature_extractor(features)
        instance_logits = self.instance_classifier(h)
        value = self.value(h)
        query_all = self.query(h)

        # Official DSMIL sorts instance scores per class and uses the top
        # instance feature as that class's query anchor.
        _, sorted_indices = torch.sort(instance_logits, dim=0, descending=True)
        critical_features = h.index_select(0, sorted_indices[0])
        critical_queries = self.query(critical_features)

        scale = math.sqrt(float(self.query_dim))
        attn = torch.softmax(torch.mm(query_all, critical_queries.transpose(0, 1)) / scale, dim=0)
        bag_repr = torch.mm(attn.transpose(0, 1), value)
        bag_logits = self.bag_classifier(bag_repr.unsqueeze(0)).view(1, self.num_classes)
        return bag_logits, attn.transpose(0, 1), instance_logits

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"DSMIL expects features with shape (B,N,D), got {tuple(features.shape)}")

        logits = []
        attentions = []
        instance_logits_for_loss = []
        for row in features:
            bag_logits, attn, inst = self._forward_one(row)
            logits.append(bag_logits)
            attentions.append(attn.mean(dim=0))
            instance_logits_for_loss.append(inst)
        self.last_attention_scores = torch.stack(attentions, dim=0).detach()
        # The detached cache remains available to diagnostic/export code.  The
        # graph-carrying cache is consumed exactly once by the loss helper below
        # so completed graphs are not retained across optimizer steps.
        self.last_instance_logits = (
            instance_logits_for_loss[0].detach()
            if len(instance_logits_for_loss) == 1
            else None
        )
        self._instance_logits_for_loss = instance_logits_for_loss
        return torch.cat(logits, dim=0)

    def consume_dual_stream_cross_entropy(
        self,
        bag_logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Return the official equal-weight DSMIL dual-stream objective.

        The official DSMIL training code optimizes both the bag prediction and
        the class-wise maximum instance prediction with equal weight.  Our
        datasets use one mutually exclusive class label, so cross-entropy is
        the protocol-consistent counterpart of the official repository's
        one-hot BCE loss.  Inference still uses ``bag_logits``, matching the
        official default where ``average=False``.

        This method consumes the private forward cache.  Failing loudly when a
        trainer calls it without a matching forward prevents the original bug,
        where the instance classifier silently received no gradient.
        """

        instance_logits = self._instance_logits_for_loss
        self._instance_logits_for_loss = None
        if instance_logits is None:
            raise RuntimeError(
                "DSMIL dual-stream loss requires one preceding forward pass; "
                "the instance-logit cache is empty or has already been consumed."
            )
        if len(instance_logits) != bag_logits.shape[0]:
            raise RuntimeError(
                "DSMIL dual-stream cache batch mismatch: "
                f"{len(instance_logits)} instance bags for {bag_logits.shape[0]} bag logits."
            )
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"unsupported DSMIL loss reduction: {reduction!r}")

        # Each class selects its own critical patch, exactly as the official
        # implementation does before applying the second classification loss.
        max_instance_logits = torch.stack(
            [logits.max(dim=0).values for logits in instance_logits],
            dim=0,
        )
        bag_loss = F.cross_entropy(
            bag_logits.float(), labels, reduction=reduction,
        )
        instance_loss = F.cross_entropy(
            max_instance_logits.float(), labels, reduction=reduction,
        )
        return 0.5 * bag_loss + 0.5 * instance_loss


def dsmil_dual_stream_cross_entropy(
    model: nn.Module,
    bag_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Apply DSMIL's dual-stream loss through an optional module wrapper."""

    base_model = model.module if hasattr(model, "module") else model
    if not isinstance(base_model, DSMILAggregator):
        raise TypeError(
            "DSMIL dual-stream loss received a non-DSMIL model: "
            f"{type(base_model).__name__}."
        )
    return base_model.consume_dual_stream_cross_entropy(
        bag_logits,
        labels,
        reduction=reduction,
    )


class OfficialNystromAttention(nn.Module):
    """Nyström attention matching the dependency path used by official TransMIL.

    This differs from the project's standard Nyström attention in two important
    ways inherited from lucidrains' package: sequence padding is placed at the
    front inside attention, and a depthwise residual convolution over V is added
    to the approximated attention output.  Keeping those choices here prevents
    the "official TransMIL" baseline from silently becoming our modified
    TransMIL/Nyström scaffold.
    """

    def __init__(
        self,
        dim: int = 512,
        *,
        dim_head: int = 64,
        heads: int = 8,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        residual: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.num_landmarks = int(num_landmarks)
        self.pinv_iterations = int(pinv_iterations)
        self.scale = self.dim_head ** -0.5
        inner_dim = self.heads * self.dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(float(dropout)))
        self.residual = bool(residual)
        if self.residual:
            self.res_conv = nn.Conv2d(
                self.heads,
                self.heads,
                kernel_size=(33, 1),
                padding=(16, 0),
                groups=self.heads,
                bias=False,
            )
        self._capture_stats = False
        self.last_stats: dict[str, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"OfficialNystromAttention expects (B,N,D), got {tuple(x.shape)}")
        bsz, n_orig, _ = x.shape
        if n_orig <= 0:
            raise ValueError("OfficialNystromAttention received an empty sequence.")

        # The lucidrains implementation uses at most N landmarks for short
        # sequences.  Without this guard, smoke tests with N < 256 would create
        # empty landmark groups.
        landmarks = min(self.num_landmarks, n_orig)
        group_len = math.ceil(n_orig / landmarks)
        pad = landmarks * group_len - n_orig
        if pad > 0:
            # Front padding mirrors lucidrains.  Slicing from the end below
            # returns the original token outputs in their original order.
            x = F.pad(x, (0, 0, pad, 0), value=0.0)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            t.view(bsz, -1, self.heads, self.dim_head).transpose(1, 2).contiguous()
            for t in qkv
        ]
        q = q * self.scale

        q_landmarks = q.view(bsz, self.heads, landmarks, group_len, self.dim_head).sum(dim=3) / group_len
        k_landmarks = k.view(bsz, self.heads, landmarks, group_len, self.dim_head).sum(dim=3) / group_len

        sim1 = torch.matmul(q, k_landmarks.transpose(-1, -2))
        sim2 = torch.matmul(q_landmarks, k_landmarks.transpose(-1, -2))
        sim3 = torch.matmul(q_landmarks, k.transpose(-1, -2))

        attn1 = torch.softmax(sim1, dim=-1)
        attn2 = torch.softmax(sim2, dim=-1)
        attn3 = torch.softmax(sim3, dim=-1)
        attn2_inv = moore_penrose_iter_inv(attn2, iters=self.pinv_iterations)

        out = torch.matmul(torch.matmul(attn1, attn2_inv), torch.matmul(attn3, v))
        if self.residual:
            out = out + self.res_conv(v)

        if self._capture_stats:
            # Store only the CLS-to-original-patch row; materializing N x N
            # would be unnecessary and too large for WSI bags.
            cls_idx = pad
            cls_kernel = torch.matmul(attn1[:, :, cls_idx:cls_idx + 1, :], attn2_inv)
            cls_to_all = torch.matmul(cls_kernel, attn3).squeeze(-2)
            self.last_stats = {
                "cls_patch_attn": cls_to_all[:, :, pad + 1: pad + n_orig].detach(),
                "num_cls_tokens": torch.tensor(1, device=x.device),
            }

        out = out.transpose(1, 2).contiguous().view(bsz, -1, self.heads * self.dim_head)
        out = self.to_out(out)
        return out[:, -n_orig:]


class OfficialTransLayer(nn.Module):
    """Official TransMIL TransLayer: pre-LayerNorm plus Nyström attention."""

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = OfficialNystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class OfficialPPEG(nn.Module):
    """PPEG block with official TransMIL attribute names and operations."""

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        bsz, _, dim = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).contiguous().view(bsz, dim, h, w)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return torch.cat((cls_token.unsqueeze(1), x), dim=1)


class OfficialTransMILAggregator(nn.Module):
    """Official TransMIL topology with project-specific input dimension."""

    def __init__(self, *, in_dim: int, num_classes: int, internal_dim: int = 512) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.internal_dim = int(internal_dim)

        self._fc1 = nn.Sequential(nn.Linear(self.in_dim, self.internal_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.internal_dim))
        self.n_classes = self.num_classes
        self.layer1 = OfficialTransLayer(dim=self.internal_dim)
        self.pos_layer = OfficialPPEG(dim=self.internal_dim)
        self.layer2 = OfficialTransLayer(dim=self.internal_dim)
        self.norm = nn.LayerNorm(self.internal_dim)
        self._fc2 = nn.Linear(self.internal_dim, self.num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"OfficialTransMIL expects features with shape (B,N,D), got {tuple(features.shape)}")
        if features.shape[1] <= 0:
            raise ValueError("OfficialTransMIL received an empty slide bag.")

        h = self._fc1(features)

        # Official TransMIL pads to a square grid by duplicating early patch
        # tokens.  The operation is deterministic and keeps the real patch
        # order unchanged before the added duplicates.
        h_grid = math.ceil(math.sqrt(h.shape[1]))
        w_grid = h_grid
        add_length = h_grid * w_grid - h.shape[1]
        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)

        cls_tokens = self.cls_token.expand(h.shape[0], -1, -1).to(h.device, h.dtype)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, h_grid, w_grid)
        h = self.layer2(h)
        h = self.norm(h)[:, 0]
        return self._fc2(h)

    @property
    def blocks(self) -> list[OfficialTransLayer]:
        """Expose a block-like list for optional attention exporters."""
        return [self.layer1, self.layer2]


def build_mil_baseline_aggregator(
    *,
    kind: BaselineKind,
    in_dim: int,
    num_classes: int,
) -> nn.Module:
    """Factory used by the shared trainers.

    Defaults are deliberately fixed here rather than threaded through existing
    SRP flags. This keeps baseline runs reproducible and prevents an
    unrelated flag such as ``--embed_dim`` from silently changing an official
    architecture.
    """

    if kind == "abmil":
        return ABMILAggregator(in_dim=in_dim, num_classes=num_classes)
    if kind == "dsmil":
        return DSMILAggregator(in_dim=in_dim, num_classes=num_classes)
    if kind == "official_transmil":
        return OfficialTransMILAggregator(in_dim=in_dim, num_classes=num_classes)
    raise ValueError(f"unknown MIL baseline kind: {kind!r}")

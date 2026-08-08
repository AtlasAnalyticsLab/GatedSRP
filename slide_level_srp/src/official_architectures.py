"""Adapters for official SPAN and Prov-GigaPath slide encoders.

The third-party implementations are loaded from optional local checkouts and
are not copied into this package:

* SPAN rows instantiate the official graph encoder through a DGL data adapter.
  Missing optional dependencies raise an explicit error instead of substituting
  a different model.
* Prov-GigaPath rows instantiate the official ``LongNetViT`` slide encoder and
  optionally insert Gated SRP after the official LongNet attention update.

The LongNet integration is written as a post-attention patch-token correction,
so the high-level Gated SRP design stays unchanged:
``r = local mean``, ``c = <y, r_hat> r_hat``, ``z = y - beta c``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import MethodType
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from slide_level_srp.src.srp_correction import PatchSRPCorrection


class OfficialArchitectureDependencyError(RuntimeError):
    """Raised when an official third-party architecture cannot run here."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _external_repo(name: str) -> Path:
    env_name = {
        "SPAN": "GATEDSRP_SPAN_ROOT",
        "prov-gigapath": "GATEDSRP_GIGAPATH_ROOT",
    }.get(name)
    if env_name and os.environ.get(env_name):
        return Path(os.environ[env_name]).expanduser().resolve()
    return _repo_root() / "external" / name


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _install_dgl_graphbolt_stub_if_needed() -> None:
    """Avoid a DGL/Torch 2.6 GraphBolt import mismatch that SPAN does not use.

    DGL 2.1 wheels ship GraphBolt binaries for older PyTorch versions and
    import GraphBolt unconditionally.  Official SPAN uses DGL graph creation,
    message passing, and ``edge_softmax`` only.  Installing a minimal in-memory
    ``dgl.graphbolt`` module before importing DGL lets those graph APIs load
    without touching site-packages or changing the DGL graph implementation.
    """

    if "dgl.graphbolt" not in sys.modules:
        stub = types.ModuleType("dgl.graphbolt")
        stub.__all__ = []
        sys.modules["dgl.graphbolt"] = stub


def _ensure_span_importable() -> None:
    repo = _external_repo("SPAN")
    if not repo.exists():
        raise OfficialArchitectureDependencyError(f"Official SPAN repo is missing: {repo}")
    if str(repo) not in sys.path:
        # SPAN imports modules as ``src.span.*``.  The local project also has a
        # top-level ``src`` package, so we put the SPAN checkout first and also
        # extend an already-imported ``src`` package below when necessary.
        sys.path.insert(0, str(repo))

    span_src = repo / "src"
    src_pkg = sys.modules.get("src")
    if src_pkg is not None and hasattr(src_pkg, "__path__"):
        path_list = list(src_pkg.__path__)
        if str(span_src) not in path_list:
            # If train_panda imported the local ``src`` package first, Python
            # would otherwise never find ``src.span``.  Extending __path__ is a
            # process-local namespace-package style adapter; it does not modify
            # the vendored SPAN repository or our project package on disk.
            src_pkg.__path__.append(str(span_src))


def _span_dgl_cuda_status() -> str:
    """Return whether official SPAN's DGL graph backend can run on this host."""

    try:
        _install_dgl_graphbolt_stub_if_needed()
        import dgl  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on host packages.
        return f"dgl_import_failed_{type(exc).__name__}"

    if not torch.cuda.is_available():
        return "ready_cpu_only"

    try:
        _install_dgl_graphbolt_stub_if_needed()
        import dgl

        device = torch.device("cuda:0")
        src = torch.tensor([0], device=device)
        dst = torch.tensor([1], device=device)
        g = dgl.graph((src, dst), num_nodes=2, device=device)
        # Touch a DGL tensor on the GPU so CPU-only builds fail here, before
        # expensive training runs are launched.
        g.ndata["x"] = torch.ones(2, 1, device=device)
        _ = g.ndata["x"].sum().item()
    except Exception as exc:  # pragma: no cover - depends on host packages.
        return f"dgl_cuda_graph_failed_{type(exc).__name__}"
    return "ready"


def _longnet_flash_attn_status() -> str | None:
    """Return a production flash-attn readiness string, if importable."""

    try:
        from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func  # noqa: F401
    except Exception:
        return None
    return "ready_flash_attn"


def _longnet_xformers_status() -> str | None:
    """Return xFormers readiness after exercising the needed CUDA API.

    LongNet needs more than ``import xformers``: it needs the
    forward-with-context API that returns an LSE tensor for branch merging.
    Running a tiny bf16 CUDA op catches ABI or kernel-dispatch mismatches before
    a manifest marks LongNet rows runnable.
    """

    if not torch.cuda.is_available():
        return None
    try:
        from xformers.ops.fmha import (
            Inputs,
            _memory_efficient_attention_forward_requires_grad,
            cutlass,
        )

        device = torch.device("cuda:0")
        q = torch.randn(1, 16, 4, 32, device=device, dtype=torch.bfloat16)
        k = torch.randn(1, 16, 4, 32, device=device, dtype=torch.bfloat16)
        v = torch.randn(1, 16, 4, 32, device=device, dtype=torch.bfloat16)
        inp = Inputs(query=q, key=k, value=v, attn_bias=None, p=0.0, scale=None)
        out, op_ctx = _memory_efficient_attention_forward_requires_grad(
            inp=inp,
            op=cutlass.FwOp,
        )
        torch.cuda.synchronize(device)
        if not torch.isfinite(out).all() or op_ctx.lse is None:
            return "xformers_backend_probe_failed_nonfinite_or_missing_lse"
    except Exception as exc:  # pragma: no cover - host/backend dependent.
        return f"xformers_backend_probe_failed_{type(exc).__name__}"
    return "ready_xformers_memory_efficient_attention"


def official_architecture_dependency_status() -> dict[str, str]:
    """Report whether each optional architecture has its runtime dependencies."""

    status: dict[str, str] = {}
    if not _external_repo("SPAN").exists():
        status["span"] = "missing_official_span_repo"
    elif not _has_module("dgl"):
        status["span"] = "missing_dgl_dependency_for_official_span"
    else:
        dgl_status = _span_dgl_cuda_status()
        status["span"] = "ready" if dgl_status in {"ready", "ready_cpu_only"} else dgl_status

    if not _external_repo("prov-gigapath").exists():
        status["longnet"] = "missing_official_prov_gigapath_repo"
    else:
        # Official LongNet depends on a memory-efficient attention kernel.
        # The local PyTorch fallback is useful only for tiny smoke tests; on
        # WSI-scale bags it materializes dense score tensors and OOMs.  Treat
        # Missing flash-attn/xFormers makes full WSI training impractical unless
        # the caller explicitly opts into the fallback for small shape checks.
        backend_status = _longnet_flash_attn_status() or _longnet_xformers_status()
        allow_torch_fallback = os.environ.get(
            "GATEDSRP_ALLOW_LONGNET_TORCH_FALLBACK", ""
        ).lower() in {"1", "true", "yes"}
        if backend_status is not None:
            status["longnet"] = backend_status
        elif allow_torch_fallback:
            status["longnet"] = "ready_torch_attention_fallback_debug_only"
        else:
            status["longnet"] = "missing_flash_attn_or_xformers_for_official_longnet"
    return status


def _ensure_gigapath_importable() -> None:
    repo = _external_repo("prov-gigapath")
    if not repo.exists():
        raise OfficialArchitectureDependencyError(
            f"Official Prov-GigaPath repo is missing: {repo}"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    inner = repo / "gigapath"
    if str(inner) not in sys.path:
        # LongNet.py imports TorchScale as a top-level ``torchscale`` package
        # after appending this same inner directory.  Adding it here lets the
        # fallback patch bind before the first forward pass.
        sys.path.insert(0, str(inner))


def _torch_flash_attention_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout: float = 0.0,
    bias: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch fallback for LongNet's flash-attention function signature.

    Official LongNet expects a callable returning ``(out, softmax_lse)`` for
    tensors shaped ``(B, L, H, D)``.  This fallback preserves that interface
    using standard PyTorch operations.  It is intentionally used only when the
    optimized CUDA extension is absent. It is intended only for small shape
    checks because it materializes dense attention scores.
    """

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise RuntimeError(
            "LongNet attention fallback expects q/k/v=(B,L,H,D); "
            f"got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    scale = float(softmax_scale) if softmax_scale is not None else q.shape[-1] ** -0.5
    # Score shape is (B,H,Lq,Lk), matching flash-attn's per-head semantics.
    scores = torch.einsum("blhd,bshd->bhls", q, k) * scale
    if bias is not None:
        if bias.ndim == 2:
            scores = scores + bias.view(1, 1, *bias.shape)
        elif bias.ndim == 3:
            scores = scores + bias.unsqueeze(1)
        elif bias.ndim == 4:
            scores = scores + bias
        else:
            raise RuntimeError(f"unsupported LongNet attention bias shape: {tuple(bias.shape)}")
    if is_causal:
        q_len, k_len = scores.shape[-2:]
        causal = torch.ones(q_len, k_len, device=scores.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal.view(1, 1, q_len, k_len), float("-inf"))

    scores_float = scores.float()
    lse = torch.logsumexp(scores_float, dim=-1).to(q.dtype)
    probs = torch.softmax(scores_float, dim=-1).to(q.dtype)
    if dropout and dropout > 0.0:
        probs = F.dropout(probs, p=float(dropout), training=torch.is_grad_enabled())
    out = torch.einsum("bhls,bshd->blhd", probs, v)
    return out, lse


class _XFormersLongNetFlashAttnFunc(torch.autograd.Function):
    """xFormers backend that matches TorchScale's flash-attention contract.

    Prov-GigaPath's vendored TorchScale code expects ``flash_attn_func`` to
    return ``(out, lse)`` for q/k/v shaped ``(B, L, H, D)``.  The public
    ``xformers.ops.memory_efficient_attention`` helper returns only ``out``;
    LongNet needs ``lse`` to merge the dilated attention branches.  This class
    mirrors the xFormers branch already present in the vendored
    ``flash_attention.py`` and exposes the exact two-value interface without
    materializing dense attention scores.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout: float = 0.0,
        bias: torch.Tensor | None = None,
        softmax_scale: float | None = None,
        is_causal: bool = False,
    ):
        from xformers.ops.fmha import (
            Context,
            Inputs,
            LowerTriangularMask,
            _memory_efficient_attention_forward_requires_grad,
            cutlass,
        )

        if is_causal:
            # Match the vendored TorchScale/xFormers path: causal masking is
            # represented by an xFormers bias object, not a dense tensor.
            if bias is not None:
                raise RuntimeError("LongNet xFormers causal attention expects bias=None")
            attn_bias = LowerTriangularMask()
        else:
            attn_bias = bias

        inp = Inputs(
            query=q,
            key=k,
            value=v,
            attn_bias=attn_bias,
            p=float(dropout),
            scale=softmax_scale,
        )
        out, op_ctx = _memory_efficient_attention_forward_requires_grad(
            inp=inp,
            op=cutlass.FwOp,
        )

        # Tensor biases must be saved through save_for_backward; structured
        # xFormers bias objects are stored separately and reconstructed in
        # backward.  This follows the vendored TorchScale implementation.
        if isinstance(inp.attn_bias, torch.Tensor):
            attn_bias_tensor = inp.attn_bias
            attn_bias_ctx = None
        else:
            attn_bias_tensor = None
            attn_bias_ctx = inp.attn_bias

        ctx.save_for_backward(inp.query, inp.key, inp.value, op_ctx.out, op_ctx.lse)
        ctx.rng_state = op_ctx.rng_state
        ctx.attn_bias_tensor = attn_bias_tensor
        ctx.attn_bias_ctx = attn_bias_ctx
        ctx.op_bw = op_ctx.op_bw if op_ctx.op_bw is not None else cutlass.BwOp
        ctx.p = inp.p
        ctx.scale = inp.scale
        return out, op_ctx.lse

    @staticmethod
    def backward(ctx, grad: torch.Tensor, dlse: torch.Tensor):
        from xformers.ops.fmha import (
            Context,
            Inputs,
            _memory_efficient_attention_backward,
        )

        query, key, value, out, lse = ctx.saved_tensors
        attn_bias = ctx.attn_bias_tensor if ctx.attn_bias_tensor is not None else ctx.attn_bias_ctx
        inp = Inputs(
            query=query,
            key=key,
            value=value,
            attn_bias=attn_bias,
            p=ctx.p,
            scale=ctx.scale,
        )
        op_ctx = Context(lse=lse, out=out, rng_state=ctx.rng_state)
        grads = _memory_efficient_attention_backward(
            ctx=op_ctx,
            inp=inp,
            grad=grad,
            op=ctx.op_bw,
        )
        return grads.dq, grads.dk, grads.dv, None, grads.db, None, None


def _xformers_flash_attention_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout: float = 0.0,
    bias: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _XFormersLongNetFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout,
        bias,
        softmax_scale,
        is_causal,
    )


def _install_longnet_attention_backend_if_needed() -> str:
    """Patch official LongNet to an available memory-efficient backend."""

    _ensure_gigapath_importable()
    module_pairs = []
    for prefix in ("gigapath.torchscale", "torchscale"):
        try:
            flash_mod = __import__(f"{prefix}.component.flash_attention", fromlist=["flash_attn_func"])
            mha_mod = __import__(f"{prefix}.component.multihead_attention", fromlist=["flash_attn_func"])
        except Exception:
            continue
        module_pairs.append((flash_mod, mha_mod))

    if module_pairs:
        existing_backend = getattr(module_pairs[0][0], "flash_attn_func", None)
        if existing_backend is _xformers_flash_attention_backend:
            return "xformers_memory_efficient_attention"
        if existing_backend is not None:
            return "flash_attn"

    if _has_module("xformers"):
        # On Ada GPUs, the vendored file tries ``flash_attn`` first and does
        # not naturally fall through to its xFormers branch.  Patching the same
        # ``flash_attn_func`` symbol keeps the official LongNet code path while
        # selecting the memory-efficient backend that is available here.
        for flash_mod, mha_mod in module_pairs:
            flash_mod.flash_attn_func = _xformers_flash_attention_backend
            mha_mod.flash_attn_func = _xformers_flash_attention_backend
        return "xformers_memory_efficient_attention"

    allow_torch_fallback = os.environ.get(
        "GATEDSRP_ALLOW_LONGNET_TORCH_FALLBACK", ""
    ).lower() in {"1", "true", "yes"}
    if allow_torch_fallback:
        # This branch is deliberately explicit: it is useful for tiny shape
        # debugging, but it computes dense score tensors and is not safe for
        # WSI-scale training jobs.
        for flash_mod, mha_mod in module_pairs:
            flash_mod.flash_attn_func = _torch_flash_attention_fallback
            mha_mod.flash_attn_func = _torch_flash_attention_fallback
        return "torch_attention_fallback_debug_only"

    raise OfficialArchitectureDependencyError(
        "Official LongNet has no memory-efficient attention backend available. "
        "Install flash-attn or xFormers, or set "
        "GATEDSRP_ALLOW_LONGNET_TORCH_FALLBACK=1 only for tiny debug runs."
    )


def _patch_longnet_segment_length_numpy_compat(LongNetViT: type[nn.Module]) -> None:
    """Make official LongNet segment-length strings robust to NumPy 2.x.

    The vendored Prov-GigaPath code does ``str(list(np_array))`` and then
    TorchScale later calls ``eval`` on that string.  With NumPy 2.x, list items
    print as ``np.int64(...)``, which fails because TorchScale's eval scope has
    no ``np``.  Returning a list of plain Python ints preserves the official
    segment schedule and fixes only the serialization incompatibility.
    """

    def get_optimal_segment_length(self, max_wsi_size: int = 262144, tile_size: int = 256) -> str:
        import numpy as np

        max_seq_len = (max_wsi_size // tile_size) ** 2
        segment_length = np.linspace(np.log2(1024), int(np.log2(max_seq_len)), 5)
        segment_length = np.power(2, segment_length).astype(int)
        return str([int(v) for v in segment_length.tolist()])

    LongNetViT.get_optimal_segment_length = get_optimal_segment_length


def _ensure_official_longnet_backend() -> None:
    status = official_architecture_dependency_status()["longnet"]
    if not status.startswith("ready"):
        raise OfficialArchitectureDependencyError(
            "Official Prov-GigaPath/LongNet integration is blocked: "
            f"{status}. Install a compatible flash-attn/xFormers backend or "
            "set GATEDSRP_ALLOW_LONGNET_TORCH_FALLBACK=1 only for tiny debug "
            "runs. The fallback is intentionally blocked for WSI-scale bags "
            "because it materializes dense attention scores."
        )


def _patch_official_longnet_layers(
    model: nn.Module,
    srp_modules: nn.ModuleList | None,
    *,
    use_activation_checkpointing: bool = True,
) -> None:
    """Insert Gated SRP after official LongNet attention in every encoder layer.

    We patch the layer instance instead of editing the vendored repository, so
    the upstream code remains auditable.  The copied control flow mirrors the
    official ``EncoderLayer.forward`` and changes only the tensor immediately
    after ``self.self_attn`` returns.
    """

    from torchscale.component.multiway_network import set_split_position

    layers = list(getattr(model.encoder, "layers", []))
    srp_module_list = list(srp_modules) if srp_modules is not None else [None] * len(layers)
    if len(layers) != len(srp_module_list):
        raise OfficialArchitectureDependencyError(
            "Official LongNet layer count does not match SRP module count: "
            f"{len(layers)} vs {len(srp_module_list)}"
        )

    def make_forward(layer: nn.Module, srp: PatchSRPCorrection | None):
        def _forward_impl(
            self,
            x,
            encoder_padding_mask,
            attn_mask=None,
            rel_pos=None,
            multiway_split_position=None,
            incremental_state=None,
            srp_context=None,
        ):
            if multiway_split_position is not None:
                # Preserve the official LongNet multiway handling.  Current
                # GigaPath slide encoders do not use multiway, but keeping the
                # branch avoids changing upstream semantics.
                assert self.args.multiway
                self.apply(set_split_position(multiway_split_position))

            if attn_mask is not None:
                attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

            residual = x
            if self.normalize_before:
                x = self.self_attn_layer_norm(x)
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                rel_pos=rel_pos,
                incremental_state=incremental_state,
            )

            # Reentrant checkpointing recomputes this layer during backward after
            # the parent aggregator forward has already returned.  The normal
            # layer attribute is therefore only a convenience for the first
            # forward; checkpointed paths must pass a captured context tuple so
            # the recomputation uses exactly the same neighbor graph and weights.
            context = srp_context if srp_context is not None else getattr(self, "_gated_srp_context", None)
            if srp is not None:
                if context is None:
                    raise RuntimeError(
                        "Official LongNet + Gated SRP layer lost its neighbor "
                        "context.  This would silently turn the checkpoint "
                        "recompute into a baseline LongNet layer, so the run is "
                        "stopped instead of producing invalid gradients."
                    )
                neighbor_index, neighbor_mask, neighbor_weight = context
                # Official LongNetViT prepends a CLS token before the encoder.
                # SRP models local patch redundancy, so only the patch-token
                # attention update is corrected and the CLS update is kept
                # exactly as the official architecture produced it.
                if x.shape[1] != neighbor_index.shape[1] + 1:
                    raise RuntimeError(
                        "LongNet SRP context is not aligned with patch tokens: "
                        f"attention_out={tuple(x.shape)} "
                        f"neighbor_index={tuple(neighbor_index.shape)}"
                    )
                patch_update = srp(
                    x[:, 1:, :],
                    neighbor_index,
                    neighbor_mask,
                    neighbor_weight=neighbor_weight,
                )
                x = torch.cat([x[:, :1, :], patch_update], dim=1)

            x = self.dropout_module(x)
            if self.drop_path is not None:
                x = self.drop_path(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.self_attn_layer_norm(x)

            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            if not self.is_moe_layer:
                x = self.ffn(x)
                l_aux = None
            else:
                x = x.transpose(0, 1)
                x, l_aux = self.moe_layer(x)
                x = x.transpose(0, 1)

            if self.drop_path is not None:
                x = self.drop_path(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, l_aux

        def forward(
            self,
            x,
            encoder_padding_mask,
            attn_mask=None,
            rel_pos=None,
            multiway_split_position=None,
            incremental_state=None,
        ):
            should_checkpoint = (
                use_activation_checkpointing
                and self.training
                and x.requires_grad
                and incremental_state is None
                and not self.is_moe_layer
            )
            if not should_checkpoint:
                srp_context = getattr(self, "_gated_srp_context", None)
                return _forward_impl(
                    self,
                    x,
                    encoder_padding_mask,
                    attn_mask=attn_mask,
                    rel_pos=rel_pos,
                    multiway_split_position=multiway_split_position,
                    incremental_state=incremental_state,
                    srp_context=srp_context,
                )

            # Official LongNet + SRP can exceed a 48 GB card on very large KGH
            # bags even when no other process is active.  This is not a design
            # change: the same official attention, same Gated SRP correction,
            # and same FFN are recomputed during backward instead of retaining
            # every intermediate activation from the forward pass.
            srp_context = getattr(self, "_gated_srp_context", None)
            def checkpointed_layer(x_tensor, padding_tensor, attn_tensor, rel_tensor):
                out, _ = _forward_impl(
                    self,
                    x_tensor,
                    padding_tensor,
                    attn_mask=attn_tensor,
                    rel_pos=rel_tensor,
                    multiway_split_position=multiway_split_position,
                    incremental_state=None,
                    srp_context=srp_context,
                )
                return out

            x = checkpoint(
                checkpointed_layer,
                x,
                encoder_padding_mask,
                attn_mask,
                rel_pos,
                # The official LongNet path uses xFormers/LongNet custom
                # autograd functions whose saved-tensor bookkeeping can differ
                # between forward and recomputation.  Reentrant checkpointing
                # avoids that non-reentrant consistency check while preserving
                # the same forward computation and RNG state.
                use_reentrant=True,
            )
            return x, None

        return MethodType(forward, layer)

    for layer, srp in zip(layers, srp_module_list):
        layer.forward = make_forward(layer, srp)


def _local_neighbor_graph_from_grid_coords(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a 3x3-minus-center neighbor graph from current SPAN grid coords.

    SPAN pads and may downsample sparse coordinates inside the encoder.  The
    input H5 neighbor graph therefore becomes stale after the first SPAN sparse
    block.  Rebuilding the graph from the current ``ins_pos`` keeps SRP aligned
    with the official SPAN token order at every layer.
    """

    if coords.ndim != 2 or coords.shape[-1] < 2:
        raise RuntimeError(f"SPAN coords must be (N,2+), got {tuple(coords.shape)}")
    xy = coords[:, :2].long()
    n_tokens = int(xy.shape[0])
    offsets = xy.new_tensor([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],           [0, 1],
        [1, -1],  [1, 0],  [1, 1],
    ])
    if n_tokens == 0:
        return xy.new_empty((1, 0, offsets.shape[0])), torch.zeros((1, 0, offsets.shape[0]), dtype=torch.bool, device=xy.device)

    y_span = (xy[:, 1].amax() - xy[:, 1].amin() + 3).clamp_min(3)
    key = xy[:, 0] * y_span + xy[:, 1]
    sorted_key, order = torch.sort(key)
    target_xy = xy[:, None, :] + offsets[None, :, :]
    target_key = target_xy[..., 0] * y_span + target_xy[..., 1]
    flat_target = target_key.reshape(-1)
    pos = torch.searchsorted(sorted_key, flat_target)
    in_bounds = pos < sorted_key.numel()
    safe_pos = pos.clamp(max=max(sorted_key.numel() - 1, 0))
    match = in_bounds & (sorted_key[safe_pos] == flat_target)
    flat_index = torch.where(match, order[safe_pos], torch.zeros_like(safe_pos))
    neighbor_index = flat_index.view(n_tokens, offsets.shape[0])
    neighbor_mask = match.view(n_tokens, offsets.shape[0])
    return neighbor_index.unsqueeze(0), neighbor_mask.unsqueeze(0)


def _patch_span_transformer_layers(model: nn.Module, srp_modules: nn.ModuleList) -> None:
    """Insert SRP after official SPAN sparse attention outputs.

    The patch targets the official slide-level SPAN transformer blocks.  It
    mirrors the upstream forward methods for the long/swin variants and changes
    only the patch-token attention branch immediately before the residual add.
    Global tokens remain unmodified because SRP models local patch redundancy.
    """

    trans_blocks: list[nn.Module] = []
    for block in getattr(model, "blocks", []):
        for module in getattr(block, "modules_dict", {}).values():
            if not isinstance(module, nn.ModuleList):
                continue
            for layer in module:
                trans_block = getattr(layer, "trans_block", None)
                if trans_block is not None:
                    trans_blocks.append(trans_block)

    if len(trans_blocks) != len(srp_modules):
        raise OfficialArchitectureDependencyError(
            "Official SPAN transformer layer count does not match SRP module count: "
            f"{len(trans_blocks)} vs {len(srp_modules)}"
        )

    def make_forward(trans_block: nn.Module, srp: PatchSRPCorrection):
        def forward(self, feat, global_feat, g1, g2, ins_pos=None):
            name = self.__class__.__name__
            if name not in {"SwinTransLayer", "LongformerTransLayer"}:
                raise OfficialArchitectureDependencyError(
                    "SPAN + Gated SRP currently supports official long/swin "
                    f"transformer blocks; got {name}."
                )

            n_local = feat.size(0)
            shortcut = torch.cat([feat, global_feat], dim=0)
            feat_norm = self.norm1(feat)
            global_norm = self.norm1(global_feat)
            attn_out = 0
            if g1 is not None:
                attn_out = attn_out + self.flexible_attention(
                    g1, feat_norm, global_norm, ins_pos,
                    alter_qkv=False, mask_type="local",
                )
            if name == "SwinTransLayer" and g2 is not None:
                attn_out = attn_out + self.flexible_attention(
                    g2, feat_norm, global_norm, ins_pos,
                    alter_qkv=True, mask_type="local",
                )
            elif name == "LongformerTransLayer" and global_norm.size(0) > 0:
                attn_out = attn_out + self.flexible_attention(
                    g2, feat_norm, global_norm, ins_pos,
                    alter_qkv=True, mask_type="global",
                )

            attn_out = self.fc_out(attn_out)
            attn_out = self.proj_drop(attn_out)
            if ins_pos is None:
                raise RuntimeError("SPAN + Gated SRP requires current sparse coordinates.")
            neighbor_index, neighbor_mask = _local_neighbor_graph_from_grid_coords(ins_pos)
            corrected_patch = srp(
                attn_out[:n_local].unsqueeze(0),
                neighbor_index.to(attn_out.device),
                neighbor_mask.to(attn_out.device),
            ).squeeze(0)
            attn_out = torch.cat([corrected_patch, attn_out[n_local:]], dim=0)

            feats = shortcut + self.drop_path(attn_out)
            feats = feats + self.drop_path(self.ffn(self.norm2(feats)))
            feat = feats[:n_local]
            global_feat = feats[n_local:]
            return feat, global_feat

        return MethodType(forward, trans_block)

    for trans_block, srp in zip(trans_blocks, srp_modules):
        trans_block.forward = make_forward(trans_block, srp)


def _span_task_from_target(target: str | None, num_classes: int) -> str:
    target_l = str(target or "").lower()
    if target_l.startswith("tcga") or target_l in {"kirc", "kirp", "luad", "stad", "ucec"}:
        return "survival"
    # TCGA survival uses four discrete hazard bins in this trainer.  This
    # fallback keeps programmatic callers safe when the dataset name is absent.
    return "survival" if int(num_classes) == 4 and target_l not in {"cam16", "cam17"} else "classification"


class OfficialSPANAggregator(nn.Module):
    """Official slide-level SPAN encoder with an optional SRP insertion."""

    def __init__(
        self,
        *,
        in_dim: int,
        num_classes: int,
        target: str | None,
        use_srp: bool,
        gate_hidden_dim: int = 16,
        delta_scale: float = 1.0,
        coord_div: int = 224,
    ) -> None:
        super().__init__()
        _ensure_span_importable()
        span_status = official_architecture_dependency_status()["span"]
        if not span_status.startswith("ready"):
            raise OfficialArchitectureDependencyError(
                f"Official SPAN integration is blocked: {span_status}."
            )

        from omegaconf import OmegaConf
        from src.span.builders import build_model_config
        from src.span.model import SPAN_Encoder, create_block
        from src.span.preprocessing import SPAN_Padder, get_sorted_indices, map_to_integer_grid

        self.use_srp = bool(use_srp)
        self.coord_div = int(coord_div)
        self.target = str(target or "")
        self._span_get_sorted_indices = get_sorted_indices
        self._span_map_to_integer_grid = map_to_integer_grid
        task = _span_task_from_target(target, num_classes)
        self.span_task = task
        self.span_survival_aggregation = "multi_addition"

        repo = _external_repo("SPAN")
        model_cfg = OmegaConf.load(repo / "configs" / "model" / "span.yaml")
        task_cfg = OmegaConf.load(repo / "configs" / f"{task}.yaml")
        if "model" in task_cfg:
            model_cfg = OmegaConf.merge(model_cfg, task_cfg.model)
        cfg = OmegaConf.create({"model": model_cfg})

        encoder_config, _, components = build_model_config(
            cfg,
            num_outputs=int(num_classes),
            in_channels=int(in_dim),
            enc_act=cfg.model.enc_act,
        )
        blocks = [
            create_block(config, idx + 1, mask_padding=True)
            for idx, config in enumerate(encoder_config)
        ]
        # Official SPAN initializes global tokens in input-feature space.  The
        # first sparse projection layer then maps both patch and global tokens
        # into the internal embed_dim.  Passing embed_dim here would make the
        # global token incompatible with the first official projection.
        self.encoder = SPAN_Encoder(
            blocks,
            embed_dim=int(in_dim),
            token_init_types=components["token_init_types"],
        )
        self.padder = SPAN_Padder(
            kernel_size=cfg.model.kernel_size,
            stride=cfg.model.stride,
            dilation=cfg.model.dilation,
            n_layers=max(len(components["encoder_config_string"]) - 1, 0),
            pad_feats=None,
        )
        embed_dim = int(components["embed_dim"])
        num_layers = len(encoder_config)

        if task == "survival":
            from tasks.vision.slide.survival.aggregators import SurvivalAggregator, SurvivalMultiAggregator

            cls_drop = float(task_cfg.survival.get("cls_drop", 0.05))
            aggr = str(task_cfg.survival.get("aggregation", "multi_addition")).lower()
            self.span_survival_aggregation = aggr
            if aggr == "multi_addition":
                self.aggregator = SurvivalMultiAggregator(
                    input_dim=embed_dim,
                    dropout=cls_drop,
                    num_layers=num_layers,
                    num_classes=int(num_classes),
                    lastnorm=False,
                    weight=False,
                )
            else:
                self.aggregator = SurvivalAggregator(
                    input_dim=embed_dim,
                    dropout=cls_drop,
                    num_classes=int(num_classes),
                )
        else:
            from tasks.vision.slide.classification.model import _create_aggregator

            self.aggregator = _create_aggregator(
                aggr_method=task_cfg.classification.aggr_method,
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_classes=int(num_classes),
                cls_drop=float(task_cfg.classification.cls_drop),
                head_act=task_cfg.classification.head_act,
                head_div=int(task_cfg.classification.head_div),
                head_norm=bool(task_cfg.classification.get("head_norm", False)),
            )

        if self.use_srp:
            trans_count = sum(
                1
                for block in self.encoder.blocks
                for module in block.modules_dict.values()
                if isinstance(module, nn.ModuleList)
                for layer in module
                if getattr(layer, "trans_block", None) is not None
            )
            self.srp_modules = nn.ModuleList([
                PatchSRPCorrection(
                    embed_dim,
                    hidden_dim=int(gate_hidden_dim),
                    delta_scale=float(delta_scale),
                )
                for _ in range(trans_count)
            ])
            _patch_span_transformer_layers(self.encoder, self.srp_modules)
        else:
            self.srp_modules = nn.ModuleList()

    def _normalize_coords(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Match official SPAN slide preprocessing: divide level-0 coordinates
        # by 224, shift to the local origin, cast onto the integer grid, then
        # sort instances by row-major grid order.  Our AtlasPatch H5s can have
        # rare near-duplicate coordinates after integer casting; DGL's sparse
        # window builder assumes one node id per grid cell, so repair only that
        # edge case with SPAN's own map_to_integer_grid utility.
        xy_float = coords[:, :2].float() / float(self.coord_div)
        xy_float = xy_float - xy_float.amin(dim=0, keepdim=True)
        xy = xy_float.long()
        if torch.unique(xy, dim=0).shape[0] != xy.shape[0]:
            xy = self._span_map_to_integer_grid(xy_float.detach().cpu()).to(
                device=coords.device,
                dtype=torch.long,
            )
        order = self._span_get_sorted_indices(xy)
        return xy[order], order

    def _forward_one(self, feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        if feats.ndim != 2 or coords.ndim != 2:
            raise RuntimeError(
                f"SPAN expects one slide as feats=(N,D), coords=(N,2+); got {tuple(feats.shape)}, {tuple(coords.shape)}"
            )
        ins_pos, order = self._normalize_coords(coords)
        feats = feats[order]
        ins_pos, feats = self.padder(ins_pos, feats)
        _, feats_list, global_feats, _, _ = self.encoder(ins_pos, feats)
        if self.span_task == "survival":
            if self.span_survival_aggregation == "multi_addition":
                return self.aggregator(global_feats)
            return self.aggregator(feats_list[-1])
        return self.aggregator(feats_list, global_feats)

    def forward(
        self,
        features: torch.Tensor,
        *,
        neighbor_index: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        neighbor_weight: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del neighbor_index, neighbor_mask, neighbor_weight
        if coords is None:
            raise RuntimeError("Official SPAN requires patch coordinates from the H5 loader.")
        if features.ndim != 3 or coords.ndim != 3:
            raise RuntimeError(
                "Official SPAN expects features=(B,N,D) and coords=(B,N,2+); "
                f"got {tuple(features.shape)} and {tuple(coords.shape)}"
            )
        logits = [self._forward_one(f, c) for f, c in zip(features, coords)]
        return torch.cat(logits, dim=0)


class OfficialGigaPathLongNetAggregator(nn.Module):
    """Official Prov-GigaPath LongNet slide encoder with optional Gated SRP."""

    def __init__(
        self,
        *,
        in_dim: int,
        embed_dim: int,
        depth: int,
        num_classes: int,
        use_srp: bool,
        drop_path_rate: float = 0.1,
        dropout: float = 0.1,
        gate_hidden_dim: int = 16,
        delta_scale: float = 1.0,
        tile_size: int = 256,
        max_wsi_size: int = 262144,
        use_activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        _ensure_gigapath_importable()
        _ensure_official_longnet_backend()
        from gigapath.slide_encoder import LongNetViT
        from gigapath.torchscale.model.LongNet import longnet_arch

        # The official factory indexes a fixed configuration registry, but an
        # unknown depth/dimension pair otherwise reaches an unbound local in
        # the third-party code. Validate the public adapter arguments here so
        # custom integrations receive the actual supported choices.
        config_name = f"LongNet_{int(depth)}_layers_{int(embed_dim)}_dim"
        if not hasattr(longnet_arch, config_name):
            supported = sorted(
                name
                for name, value in vars(longnet_arch).items()
                if name.startswith("LongNet_")
                and name.endswith("_dim")
                and isinstance(value, dict)
            )
            raise OfficialArchitectureDependencyError(
                f"Official Prov-GigaPath has no {config_name!r} configuration. "
                f"Supported configurations: {', '.join(supported)}"
            )

        _patch_longnet_segment_length_numpy_compat(LongNetViT)
        self.attention_backend = _install_longnet_attention_backend_if_needed()

        self.use_srp = bool(use_srp)
        self.use_activation_checkpointing = bool(use_activation_checkpointing)
        self.encoder = LongNetViT(
            in_chans=int(in_dim),
            embed_dim=int(embed_dim),
            depth=int(depth),
            tile_size=int(tile_size),
            max_wsi_size=int(max_wsi_size),
            global_pool=False,
            dropout=float(dropout),
            drop_path_rate=float(drop_path_rate),
        )
        self.head = nn.Linear(int(embed_dim), int(num_classes))

        if self.use_srp:
            self.srp_modules = nn.ModuleList([
                PatchSRPCorrection(
                    int(embed_dim),
                    hidden_dim=int(gate_hidden_dim),
                    delta_scale=float(delta_scale),
                )
                for _ in range(int(depth))
            ])
            _patch_official_longnet_layers(
                self.encoder,
                self.srp_modules,
                use_activation_checkpointing=self.use_activation_checkpointing,
            )
        else:
            self.srp_modules = nn.ModuleList()
            if self.use_activation_checkpointing:
                _patch_official_longnet_layers(
                    self.encoder,
                    None,
                    use_activation_checkpointing=self.use_activation_checkpointing,
                )

    def _set_srp_context(
        self,
        neighbor_index: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None,
        neighbor_weight: torch.Tensor | None,
    ) -> None:
        if not self.use_srp:
            return
        if neighbor_index is None or neighbor_mask is None:
            raise RuntimeError("Official LongNet + Gated SRP requires neighbor_index and neighbor_mask.")
        for layer in self.encoder.encoder.layers:
            layer._gated_srp_context = (neighbor_index, neighbor_mask, neighbor_weight)

    def _clear_srp_context(self) -> None:
        if not self.use_srp:
            return
        for layer in self.encoder.encoder.layers:
            if hasattr(layer, "_gated_srp_context"):
                delattr(layer, "_gated_srp_context")

    def _validate_coords(self, coords: torch.Tensor) -> None:
        """Fail early if slide coordinates cannot index GigaPath positions."""

        if coords.shape[-1] < 2:
            raise RuntimeError(
                "Official LongNet requires at least x/y coordinates; "
                f"got coords shape {tuple(coords.shape)}"
            )
        if not torch.isfinite(coords[..., :2]).all():
            raise RuntimeError("Official LongNet received non-finite patch coordinates.")

        # GigaPath converts level-0 pixel coordinates to a fixed 1000 x 1000
        # positional table.  A negative or too-large coordinate would otherwise
        # fail later as an opaque indexing error inside the vendored encoder.
        grid = torch.floor(coords[..., :2] / float(self.encoder.tile_size)).long()
        if bool((grid < 0).any().item()):
            bad = grid[grid < 0][0].item()
            raise RuntimeError(
                "Official LongNet received negative grid coordinates after "
                f"tile scaling; first bad value={bad}."
            )
        pos = grid[..., 0] * int(self.encoder.slide_ngrids) + grid[..., 1] + 1
        max_pos = int(self.encoder.pos_embed.shape[1]) - 1
        if bool((pos > max_pos).any().item()):
            raise RuntimeError(
                "Official LongNet coordinate positional index exceeds the "
                f"fixed table: max_pos={int(pos.max().item())}, table_max={max_pos}. "
                "Check tile_size/max_wsi_size/coordinate level before launching."
            )

    def forward(
        self,
        features: torch.Tensor,
        *,
        neighbor_index: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        neighbor_weight: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if coords is None:
            raise RuntimeError("Official LongNet requires patch coordinates for coords_to_pos().")
        if features.ndim != 3 or coords.ndim != 3:
            raise RuntimeError(
                "Official LongNet expects features=(B,N,D) and coords=(B,N,2); "
                f"got {tuple(features.shape)} and {tuple(coords.shape)}"
            )
        self._validate_coords(coords)
        self._set_srp_context(neighbor_index, neighbor_mask, neighbor_weight)
        try:
            # ``all_layer_embed=False`` keeps the official final CLS readout.
            # The classifier is task-local rather than loaded from a slide checkpoint.
            cls = self.encoder(features, coords, all_layer_embed=False)[0]
        finally:
            self._clear_srp_context()
        return self.head(cls)


def build_official_span_unavailable(*, target: str) -> nn.Module:
    """Raise an explicit SPAN blocker instead of falling back to a proxy."""

    status = official_architecture_dependency_status()["span"]
    raise OfficialArchitectureDependencyError(
        f"Official SPAN {target} integration is blocked: {status}. "
        "Use the official SPAN-MIL/slide-survival path after DGL and the "
        "SPAN H5 adapter are available; SPAN-UNet is not the slide-level "
        "classification/survival target used by this package."
    )

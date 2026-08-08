"""Small synchronized runtime and CUDA-memory profiler for WSI trainers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch


class RuntimeProfiler:
    """Accumulate phase throughput without changing the training schedule."""

    def __init__(self, *, enabled: bool, device: torch.device) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self._phase_start: float | None = None
        self._phase_name: str | None = None
        self.records: dict[str, list[dict[str, float | int]]] = {}
        if self.enabled and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _synchronize(self) -> None:
        # CUDA kernels are asynchronous; explicit synchronization is required
        # for phase-level throughput to describe completed model work.
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self, phase: str) -> None:
        if not self.enabled:
            return
        if self._phase_start is not None:
            raise RuntimeError(f"runtime phase {self._phase_name!r} is still active")
        self._synchronize()
        self._phase_name = str(phase)
        self._phase_start = time.perf_counter()

    def stop(self, *, n_slides: int) -> None:
        if not self.enabled:
            return
        if self._phase_start is None or self._phase_name is None:
            raise RuntimeError("no runtime phase is active")
        self._synchronize()
        seconds = time.perf_counter() - self._phase_start
        self.records.setdefault(self._phase_name, []).append(
            {
                "seconds": float(seconds),
                "slides": int(n_slides),
                "slides_per_second": float(n_slides / max(seconds, 1.0e-12)),
            }
        )
        self._phase_start = None
        self._phase_name = None

    def summary(self) -> dict:
        if self._phase_start is not None:
            raise RuntimeError(f"runtime phase {self._phase_name!r} was not stopped")
        phases = {}
        for phase, rows in self.records.items():
            phases[phase] = {
                "measurements": rows,
                "seconds_mean": float(sum(float(r["seconds"]) for r in rows) / len(rows)),
                "slides_per_second_mean": float(
                    sum(float(r["slides_per_second"]) for r in rows) / len(rows)
                ),
            }
        memory = {"peak_allocated_mb": None, "peak_reserved_mb": None}
        if self.enabled and self.device.type == "cuda":
            memory = {
                "peak_allocated_mb": float(
                    torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                ),
                "peak_reserved_mb": float(
                    torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
                ),
            }
        return {
            "enabled": self.enabled,
            "device": str(self.device),
            "memory": memory,
            "phases": phases,
        }

    def write(self, path: str | Path) -> None:
        if not self.enabled:
            return
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


__all__ = ["RuntimeProfiler"]

"""Synchronized timing/memory helpers for CPU and CUDA runs."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


def synchronize(device=None) -> None:
    if torch.cuda.is_available() and (device is None or torch.device(device).type == "cuda"):
        torch.cuda.synchronize(device)


@dataclass
class TimingResult:
    latency_ms: float
    peak_memory_mb: float
    device_type: str


class SynchronizedTimer:
    def __init__(self, device=None):
        self.device = device
        self._start = None
        self._cuda_start = None
        self._cuda_end = None
        self._cuda_stream = None

    def start(self, *, reset_memory: bool = True) -> None:
        device = torch.device(self.device) if self.device is not None else None
        synchronize(device)
        self._start = time.perf_counter()
        if device is not None and device.type == "cuda" and torch.cuda.is_available():
            if reset_memory:
                torch.cuda.reset_peak_memory_stats(device)
            self._cuda_start = torch.cuda.Event(enable_timing=True)
            self._cuda_end = torch.cuda.Event(enable_timing=True)
            self._cuda_stream = torch.cuda.current_stream(device)
            self._cuda_start.record(self._cuda_stream)

    def stop(self) -> TimingResult:
        device = torch.device(self.device) if self.device is not None else torch.device("cpu")
        if self._cuda_end is not None:
            self._cuda_end.record(self._cuda_stream)
            synchronize(device)
            latency_ms = float(self._cuda_start.elapsed_time(self._cuda_end))
            peak = float(torch.cuda.max_memory_allocated(device) / 1024**2)
        else:
            synchronize(device)
            latency_ms = float((time.perf_counter() - self._start) * 1000.0)
            peak = 0.0
        return TimingResult(latency_ms, peak, device.type)


def estimate_attention_ratio(original_k_len: int, compressed_k_len: int) -> float:
    if original_k_len <= 0:
        raise ValueError("original_k_len must be positive")
    if not 0 <= compressed_k_len <= original_k_len:
        raise ValueError("compressed_k_len must be within the original length")
    return float(compressed_k_len / original_k_len)

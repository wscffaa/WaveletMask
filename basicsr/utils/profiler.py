"""轻量级训练/前向剖析工具（时间 + 显存）。

使用方式：
- 在需要的代码片段加上 `from basicsr.utils.profiler import profile, profiler`，并用：
    `with profile('G forward'):` 包裹待计时代码。
- 在每个 iter 结束时调用 `profiler.report_and_reset(iter_idx)` 打印并清零统计。
- 显存记录自动追踪峰值，无需额外代码。

可通过环境变量 `BASICOFR_PROFILE=1` 或 `MAMBAOFR_PROFILE=1` 启用。
输出格式：[PROFILE][iter N] time: X ms | mem_peak: Y MB | mem_alloc: Z MB
"""

from __future__ import annotations

import os
import time
import contextlib
from typing import Dict

import torch


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer(contextlib.AbstractContextManager):
    """计时上下文管理器（GPU 同步）。"""

    def __init__(self, name: str, collector: "IterationProfiler") -> None:
        self.name = name
        self.collector = collector
        self._start = 0.0

    def __enter__(self):
        _cuda_sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        _cuda_sync()
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self.collector.add_time(self.name, elapsed_ms)
        return False


class IterationProfiler:
    def __init__(self) -> None:
        self.enabled: bool = os.environ.get('BASICOFR_PROFILE', os.environ.get('MAMBAOFR_PROFILE', '0')) in ('1', 'true', 'True')
        self.stats: Dict[str, float] = {}
        self._mem_tracking: bool = False

    def enable(self, flag: bool = True):
        self.enabled = flag

    def reset(self):
        self.stats.clear()

    def add_time(self, name: str, ms: float):
        if not self.enabled:
            return
        self.stats[name] = self.stats.get(name, 0.0) + float(ms)

    def timer(self, name: str):
        if not self.enabled:
            return contextlib.nullcontext()
        return Timer(name, self)

    def start_memory_tracking(self):
        """在每个 iter 开始时调用，重置显存峰值统计"""
        if not self.enabled or not torch.cuda.is_available():
            return
        self._mem_tracking = True
        torch.cuda.reset_peak_memory_stats()

    def get_memory_stats(self) -> Dict[str, float]:
        """获取当前显存统计 (MB)"""
        if not torch.cuda.is_available():
            return {}
        return {
            "mem_peak": torch.cuda.max_memory_allocated() / 1024 / 1024,
            "mem_alloc": torch.cuda.memory_allocated() / 1024 / 1024,
            "mem_reserved": torch.cuda.memory_reserved() / 1024 / 1024,
        }

    def report_and_reset(self, iter_idx: int | None = None):
        if not self.enabled or not self.stats:
            self.reset()
            return
        # 构造单行打印，便于日志对齐与 grep
        prefix = f"[PROFILE][iter {iter_idx}]" if iter_idx is not None else "[PROFILE]"
        total = sum(self.stats.values())
        parts = [f"{k}: {v:.2f} ms" for k, v in self.stats.items()]
        parts.append(f"total: {total:.2f} ms")

        # 添加显存统计
        if self._mem_tracking and torch.cuda.is_available():
            mem = self.get_memory_stats()
            parts.append(f"mem_peak: {mem['mem_peak']:.0f} MB")
            parts.append(f"mem_alloc: {mem['mem_alloc']:.0f} MB")

        print(prefix, " | ", " | ".join(parts), flush=True)
        self.reset()


# 全局实例与简化入口
profiler = IterationProfiler()


def profile(name: str):
    return profiler.timer(name)


def start_memory_tracking():
    """在每个 iter 开始时调用，启用显存峰值追踪"""
    profiler.start_memory_tracking()


def get_memory_mb() -> float:
    """获取当前峰值显存 (MB)，用于评估实验"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def is_profiling_enabled() -> bool:
    return profiler.enabled


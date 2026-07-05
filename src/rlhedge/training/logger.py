"""Lightweight metric logger that writes to stdout and a JSON-lines file."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class MetricLogger:
    """Accumulates scalar metrics per episode/step and periodically logs them.

    Usage::

        logger = MetricLogger(log_dir="runs/exp1")
        for episode in range(n_episodes):
            ...
            logger.record("reward", ep_reward)
            logger.record("pnl", ep_pnl)
            logger.dump(step=episode)
    """

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        print_freq: int = 1,
    ) -> None:
        self.print_freq = print_freq
        self._scalars: Dict[str, List[float]] = defaultdict(list)
        self._step = 0
        self._start_time = time.time()

        self._log_file = None
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = open(log_dir / "metrics.jsonl", "a", buffering=1)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, key: str, value: float) -> None:
        """Append a scalar observation for *key*."""
        self._scalars[key].append(float(value))

    def record_dict(self, d: Dict[str, Any]) -> None:
        """Convenience wrapper to record multiple keys at once."""
        for k, v in d.items():
            self.record(k, v)

    # ------------------------------------------------------------------
    # Dumping
    # ------------------------------------------------------------------

    def dump(self, step: Optional[int] = None) -> Dict[str, float]:
        """Compute mean of buffered scalars, log them, and clear the buffer.

        Returns the dict of means so callers can inspect values.
        """
        if step is not None:
            self._step = step
        else:
            self._step += 1

        means: Dict[str, float] = {
            k: float(sum(v) / len(v)) for k, v in self._scalars.items() if v
        }
        elapsed = time.time() - self._start_time

        if self._step % self.print_freq == 0:
            parts = [f"step={self._step}", f"elapsed={elapsed:.1f}s"]
            parts += [f"{k}={v:.4f}" for k, v in sorted(means.items())]
            print("  ".join(parts))

        if self._log_file is not None:
            row = {"step": self._step, "elapsed": elapsed, **means}
            self._log_file.write(json.dumps(row) + "\n")

        self._scalars.clear()
        return means

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the log file (if any)."""
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

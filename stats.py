"""
Inference timing, reward, and rollout statistics utilities.

This module records:
    - Per-step inference timings
    - Per-episode rewards
    - Episode durations and step counts
    - SQLite step logs for post-processing and analysis

This file was adapted from:

https://github.com/real-world-drl/esp-dl-quant-icra2026

Original project licensed under the MIT License.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


log = logging.getLogger(__name__)


@dataclass
class EpisodeStats:
    episode_no: int
    reward: float
    steps: int
    wall_seconds: float
    inference_times_us: List[int] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.steps / self.wall_seconds if self.wall_seconds > 0 else float('nan')

    @property
    def mean_inference_us(self) -> float:
        return statistics.fmean(self.inference_times_us) if self.inference_times_us else float('nan')

    @property
    def stdev_inference_us(self) -> float:
        n = len(self.inference_times_us)
        if n < 2:
            return float('nan')
        return statistics.stdev(self.inference_times_us)


@dataclass
class ComparatorStepInfo:
    """
    Per-step comparator decision information for logging.
    """

    step: int
    current_policy: str
    next_policy: str
    switch_committed: bool

    candidate_count: int
    best_idx: Optional[int]
    best_score: Optional[float]

    query_local_reward_mean: Optional[float]
    best_candidate_local_reward_mean: Optional[float]
    best_candidate_policy: Optional[str]
    best_candidate_condition: Optional[str]


class InferenceStats:
    """Collects per-episode and aggregate statistics across an evaluation run."""

    def __init__(self, *, output_dir: Optional[str | Path] = None, sqlite_name: str = 'inference_times.db') -> None:
        self.episodes: List[EpisodeStats] = []
        self._output_dir = Path(output_dir) if output_dir else None
        self._sqlite_path = (self._output_dir / sqlite_name) if self._output_dir else None
        self._sqlite: Optional[sqlite3.Connection] = None

        if self._sqlite_path:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._sqlite = sqlite3.connect(self._sqlite_path)
            self._sqlite.execute(
                'CREATE TABLE IF NOT EXISTS inference_times ('
                ' id INTEGER PRIMARY KEY AUTOINCREMENT,'
                ' episode_no INTEGER,'
                ' step INTEGER,'
                ' inference_time_us INTEGER)'
            )
            self._sqlite.execute(
                'CREATE TABLE IF NOT EXISTS episodes ('
                ' id INTEGER PRIMARY KEY AUTOINCREMENT,'
                ' episode_no INTEGER,'
                ' reward REAL,'
                ' steps INTEGER,'
                ' wall_seconds REAL)'
            )
            self._sqlite.execute(
                "CREATE TABLE IF NOT EXISTS comparator_steps ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " episode_no INTEGER,"
                " step INTEGER,"
                " current_policy TEXT,"
                " next_policy TEXT,"
                " switch_committed INTEGER,"
                " candidate_count INTEGER,"
                " best_idx INTEGER,"
                " best_score REAL,"
                " query_local_reward_mean REAL,"
                " best_candidate_local_reward_mean REAL,"
                " best_candidate_policy TEXT,"
                " best_candidate_condition TEXT)"
            )
            self._sqlite.commit()

    # -------------------------------------------------- recording -----
    def record_episode(self, ep: EpisodeStats) -> None:
        self.episodes.append(ep)
        if self._sqlite is None:
            return
        cur = self._sqlite.cursor()
        cur.executemany(
            'INSERT INTO inference_times (episode_no, step, inference_time_us) VALUES (?, ?, ?)',
            [(ep.episode_no, i, t) for i, t in enumerate(ep.inference_times_us)],
        )
        cur.execute(
            'INSERT INTO episodes (episode_no, reward, steps, wall_seconds) VALUES (?, ?, ?, ?)',
            (ep.episode_no, ep.reward, ep.steps, ep.wall_seconds),
        )
        self._sqlite.commit()


    def record_comparator_step(
        self,
        *,
        episode_no: int,
        step_info: ComparatorStepInfo,
    ) -> None:
        """
        Record one comparator decision/debug row.
        """

        if self._sqlite is None:
            return

        self._sqlite.execute(
            "INSERT INTO comparator_steps ("
            " episode_no,"
            " step,"
            " current_policy,"
            " next_policy,"
            " switch_committed,"
            " candidate_count,"
            " best_idx,"
            " best_score,"
            " query_local_reward_mean,"
            " best_candidate_local_reward_mean,"
            " best_candidate_policy,"
            " best_candidate_condition"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(episode_no),
                int(step_info.step),
                str(step_info.current_policy),
                str(step_info.next_policy),
                int(bool(step_info.switch_committed)),
                int(step_info.candidate_count),
                step_info.best_idx,
                step_info.best_score,
                step_info.query_local_reward_mean,
                step_info.best_candidate_local_reward_mean,
                step_info.best_candidate_policy,
                step_info.best_candidate_condition,
            ),
        )

    def close(self) -> None:
        if self._sqlite is not None:
            self._sqlite.close()
            self._sqlite = None

    # -------------------------------------------------- aggregates ----
    @property
    def all_inference_times_us(self) -> List[int]:
        out: List[int] = []
        for ep in self.episodes:
            out.extend(ep.inference_times_us)
        return out

    def summary(self) -> dict:
        rewards = [ep.reward for ep in self.episodes]
        steps = [ep.steps for ep in self.episodes]
        all_inf = self.all_inference_times_us

        def _mean(xs: Iterable[float]) -> float:
            xs = list(xs)
            return statistics.fmean(xs) if xs else float('nan')

        def _stdev(xs: Iterable[float]) -> float:
            xs = list(xs)
            return statistics.stdev(xs) if len(xs) >= 2 else float('nan')

        return {
            'episodes': len(self.episodes),
            'mean_reward': _mean(rewards),
            'stdev_reward': _stdev(rewards),
            'mean_steps': _mean(steps),
            'mean_inference_us': _mean(all_inf),
            'stdev_inference_us': _stdev(all_inf),
            'min_inference_us': min(all_inf) if all_inf else float('nan'),
            'max_inference_us': max(all_inf) if all_inf else float('nan'),
            'p50_inference_us': _percentile(all_inf, 50),
            'p95_inference_us': _percentile(all_inf, 95),
            'p99_inference_us': _percentile(all_inf, 99),
        }

    def print_summary(self) -> None:
        s = self.summary()
        print('=== Per-episode results ===')
        print(f'{"#":>3}  {"reward":>10}  {"steps":>6}  {"wall_s":>8}  {"mean_us":>9}  {"std_us":>9}  {"fps":>7}')
        for i, ep in enumerate(self.episodes, 1):
            print(f'{i:>3}  {ep.reward:>10.3f}  {ep.steps:>6d}  {ep.wall_seconds:>8.2f}  '
                  f'{ep.mean_inference_us:>9.1f}  {ep.stdev_inference_us:>9.1f}  {ep.fps:>7.2f}')
        print()
        print('=== Aggregate ===')
        print(f'  mean reward: {s["mean_reward"]:.3f} +/- {s["stdev_reward"]:.3f}')
        print(f'  mean steps:  {s["mean_steps"]:.1f}')
        print(f'  inference:   mean {s["mean_inference_us"]:.1f} us, '
              f'stdev {s["stdev_inference_us"]:.1f}, '
              f'min {s["min_inference_us"]}, max {s["max_inference_us"]}')
        print(f'  inference percentiles: p50={s["p50_inference_us"]:.1f}, '
              f'p95={s["p95_inference_us"]:.1f}, p99={s["p99_inference_us"]:.1f}')
        if self._sqlite_path:
            print(f'  sqlite log:  {self._sqlite_path}')


def _percentile(xs: List[int], q: float) -> float:
    if not xs:
        return float('nan')
    s = sorted(xs)
    idx = (len(s) - 1) * q / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(s[int(idx)])
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)

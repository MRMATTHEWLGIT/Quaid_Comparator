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
    can_switch: bool

    candidate_count: int

    query_local_reward_mean: Optional[float]

    query_umap_x: Optional[float]
    query_umap_y: Optional[float]

    candidate_indices_json: Optional[str] = None
    policy_vote_counts_json: Optional[str] = None
    selected_policy_count: Optional[int] = None
    selected_policy_fraction: Optional[float] = None
    candidate_filter_counts_json: Optional[str] = None

    # Second policy statistics
    second_policy: Optional[str] = None
    second_policy_count: Optional[int] = None
    second_policy_fraction: Optional[float] = None
    vote_margin: Optional[float] = None

    # Reward statistics
    reward_distance: Optional[float] = None
    reward_roll: Optional[float] = None
    reward_current: Optional[float] = None
    reward_yaw: Optional[float] = None
    reward_pitch: Optional[float] = None
    reward_action_smoothness: Optional[float] = None
    step_reward_total: Optional[float] = None
    episode_reward_total: Optional[float] = None

    comparator_ran: Optional[bool] = False


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
                " can_switch INTEGER,"
                " candidate_count INTEGER,"
                " candidate_filter_counts_json TEXT,"
                " candidate_indices_json TEXT,"
                " policy_vote_counts_json TEXT,"
                " selected_policy_count INTEGER,"
                " selected_policy_fraction REAL,"
                " second_policy TEXT,"
                " second_policy_count INTEGER,"
                " second_policy_fraction REAL,"
                " vote_margin REAL,"
                " reward_yaw REAL,"
                " reward_pitch REAL,"
                " reward_action_smoothness REAL,"
                " query_local_reward_mean REAL,"
                " query_umap_x REAL,"
                " query_umap_y REAL,"
                " reward_distance REAL,"
                " reward_roll REAL,"
                " reward_current REAL,"
                " step_reward_total REAL,"
                " episode_reward_total REAL,"
                " comparator_ran INTEGER)"
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
            " can_switch,"
            " candidate_count,"
            " candidate_filter_counts_json,"
            " candidate_indices_json,"
            " policy_vote_counts_json,"
            " selected_policy_count,"
            " selected_policy_fraction,"
            " second_policy,"
            " second_policy_count,"
            " second_policy_fraction,"
            " vote_margin,"
            " reward_yaw,"
            " reward_pitch,"
            " reward_action_smoothness,"
            " query_local_reward_mean,"
            " query_umap_x,"
            " query_umap_y,"
            " reward_distance,"
            " reward_roll,"
            " reward_current,"
            " step_reward_total,"
            " episode_reward_total,"
            "comparator_ran"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(episode_no),
                int(step_info.step),
                str(step_info.current_policy),
                str(step_info.next_policy),
                int(bool(step_info.switch_committed)),
                int(bool(step_info.can_switch)),
                int(step_info.candidate_count),
                step_info.candidate_filter_counts_json,
                step_info.candidate_indices_json,
                step_info.policy_vote_counts_json,
                step_info.selected_policy_count,
                step_info.selected_policy_fraction,
                step_info.second_policy,
                step_info.second_policy_count,
                step_info.second_policy_fraction,
                step_info.vote_margin,
                step_info.reward_yaw,
                step_info.reward_pitch,
                step_info.reward_action_smoothness,
                step_info.query_local_reward_mean,
                step_info.query_umap_x,
                step_info.query_umap_y,
                step_info.reward_distance,
                step_info.reward_roll,
                step_info.reward_current,
                step_info.step_reward_total,
                step_info.episode_reward_total,
                int(bool(step_info.comparator_ran))
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

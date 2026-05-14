"""Reward computation for QuaidEnv.

Mirrors ``QuaidEnv::calculate_reward`` (QuaidEnv.cpp:315-360) term-for-term.
Each term is multiplied by its config-supplied ratio; ratios of zero cause
the term to drop out cleanly.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quaid_env.config import RewardSettings
from quaid_env.quaid_data import QuaidData


@dataclass
class RewardBreakdown:
    distance: float = 0.0
    speed: float = 0.0
    yaw_delta: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    z: float = 0.0
    current: float = 0.0
    current_front_left: float = 0.0
    current_front_right: float = 0.0
    current_back_left: float = 0.0
    current_back_right: float = 0.0
    acc_x: float = 0.0
    acc_y: float = 0.0
    acc_z: float = 0.0
    action_smoothness: float = 0.0
    total: float = 0.0


def _action_smoothness(history: Sequence[np.ndarray]) -> float:
    """Sum of squared first + second derivatives across action dims.

    Replicates QuaidEnv.cpp:321-326. Only computed when at least 3 actions
    are buffered.
    """
    if len(history) < 3:
        return 0.0
    a_t, a_tm1, a_tm2 = history[-1], history[-2], history[-3]
    d1 = a_t - a_tm1
    d2 = a_t - 2.0 * a_tm1 + a_tm2
    return float(np.sum(d1 * d1 + d2 * d2))


def compute_reward(
    *,
    data: QuaidData,
    distance: float,
    yaw_delta: float,
    settings: RewardSettings,
    action_history: Sequence[np.ndarray],
) -> RewardBreakdown:
    """Compute the reward breakdown for the current step.

    ``distance`` and ``yaw_delta`` are the unnormalised quantities the env
    has already computed (from successive QuaidData snapshots). ``data`` is
    the snapshot at the END of this step.
    """
    cutoff = settings.current_cut_off

    if settings.reward_type == 'euclidean':
        signed_distance = abs(distance)
    else:
        signed_distance = distance

    speed = math.exp(-10.0 * abs(settings.target_distance_per_step - distance)) * settings.speed_reward_ratio

    def _per_leg(value: float) -> float:
        return abs(value) * settings.current_per_leg_ratio if abs(value) > cutoff / 4.0 else 0.0

    breakdown = RewardBreakdown(
        distance=signed_distance * settings.distance_reward_ratio,
        speed=speed,
        yaw_delta=yaw_delta * settings.yaw_delta_reward_ratio,
        yaw=math.exp(-10.0 * abs(data.yaw)) * settings.yaw_reward_ratio,
        pitch=abs(data.pitch) * settings.pitch_reward_ratio,
        roll=abs(data.roll) * settings.roll_reward_ratio,
        z=abs(settings.z_center - data.position_z) * settings.z_reward_ratio,
        current_front_left=_per_leg(data.current_front_left),
        current_front_right=_per_leg(data.current_front_right),
        current_back_left=_per_leg(data.current_back_left),
        current_back_right=_per_leg(data.current_back_right),
        current=math.exp(-1.5 * abs(cutoff - data.current)) * settings.current_reward_ratio,
        acc_z=abs(data.acc_z - 9.8) * settings.acc_z_reward_ratio,
        acc_x=abs(data.acc_x) * settings.acc_x_reward_ratio,
        acc_y=abs(data.acc_y) * settings.acc_y_reward_ratio,
        action_smoothness=_action_smoothness(action_history) * settings.action_smoothness_reward_ratio,
    )

    breakdown.total = (
        breakdown.distance + breakdown.speed + breakdown.yaw_delta + breakdown.yaw
        + breakdown.pitch + breakdown.roll + breakdown.z
        + breakdown.current_front_left + breakdown.current_front_right
        + breakdown.current_back_left + breakdown.current_back_right
        + breakdown.current + breakdown.acc_z + breakdown.acc_x + breakdown.acc_y
        + breakdown.action_smoothness
    )
    return breakdown

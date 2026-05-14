"""Random world-frame yaw rotation ("theta adjustment").

Mirrors ``QuaidEnv::adjust_theta`` (QuaidEnv.cpp:380-435). Theta is a virtual
frame rotation applied to the mocap yaw — the policy observes
``yaw - theta`` rather than the raw mocap yaw, which makes the policy
direction-agnostic. The rotation is sampled either at episode reset or every
``[reset_theta_min_steps, reset_theta_max_steps]`` frames during the episode.

The QuaidSIM-v4 config sets ``adjust_theta_distribution: zero`` so theta is
pinned to zero, but the full machinery is implemented for other configs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from quaid_env.config import RobotSettings
from quaid_env.quaid_data import SharedQuaidData


DEG_PER_RAD = 180.0 / math.pi  # matches QuaidUtils::DEG_TO_RAD (the C++ name


@dataclass
class ThetaUpdate:
    old_theta: float
    yaw: float
    mean_yaw: float
    random_theta: float
    new_theta: float


class ThetaController:
    """Owns RNG state and theta-reset scheduling.

    The env asks ``maybe_adjust(frame, mean_yaw, shared_data, send_message)``
    each step; this class returns either ``None`` or a populated
    ``ThetaUpdate`` along with the message string the env should publish.
    """

    def __init__(self, robot: RobotSettings, rng: random.Random | None = None) -> None:
        self._robot = robot
        self._rng = rng or random.Random()
        self._next_reset_frame = 100  # matches QuaidEnv.h:81 default

    @property
    def next_reset_frame(self) -> int:
        return self._next_reset_frame

    def schedule_next(self, frame: int) -> None:
        rng = self._rng.randint(self._robot.reset_theta_min_steps, self._robot.reset_theta_max_steps)
        self._next_reset_frame = frame + rng

    def _sample_random_theta(self, mean_yaw: float) -> float:
        """Reproduce QuaidEnv.cpp:399-409 — pick a theta offset based on the
        configured distribution, with bounds derived from the running
        ``mean_yaw`` so the sampler tightens as the robot stabilises.
        """
        dist = self._robot.adjust_theta_distribution
        yaw_end = self._robot.yaw_end_episode

        if dist == 'normal':
            sigma = max(yaw_end - abs(mean_yaw), 0.0)
            sample = self._rng.gauss(0.0, sigma) if sigma > 0 else 0.0
            sample = max(-yaw_end, min(yaw_end, sample)) / 2.0
            return sample
        if dist == 'uniform':
            limit = yaw_end * math.exp(-2.5 * abs(mean_yaw))
            return self._rng.uniform(-limit, limit)
        # 'zero' or anything unrecognised -> pin to zero
        return 0.0

    def adjust(self, shared: SharedQuaidData, mean_yaw: float, frame: int, episode_no: int, step: int) -> tuple[ThetaUpdate, str]:
        """Apply a theta rotation to ``shared`` in place; return the update
        record and the MQTT control message the env should publish.
        """
        with shared.with_lock() as data:
            old_theta = data.theta
            yaw = data.yaw
            random_theta = self._sample_random_theta(mean_yaw)
            new_theta = math.fmod(data.no_rotation_yaw + random_theta, 2 * math.pi)
            data.theta = new_theta
            data.pivot_x = data.no_rotation_x
            data.pivot_y = data.no_rotation_y

        update = ThetaUpdate(
            old_theta=old_theta,
            yaw=yaw,
            mean_yaw=mean_yaw,
            random_theta=random_theta,
            new_theta=new_theta,
        )
        # The robot expects a "pTH<degrees>" control message so it can rotate
        # its own mocap reference accordingly (QuaidEnv.cpp:431-433).
        msg = f'pTH{int(new_theta * DEG_PER_RAD)}'
        return update, msg

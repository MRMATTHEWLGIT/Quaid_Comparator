"""Gymnasium env that ties the MQTT controller, observation builder, and
reward computation together.

This is the Python counterpart to ``QuaidEnv`` in the C++ repo. It does NOT
load any ML model — the inference loop is the consumer's responsibility, so
this env is reusable by any policy framework that speaks the gymnasium API.

Lifecycle::

    settings = load_settings('quaid-icra-sim.yaml')
    env = QuaidEnv(settings)
    env.connect()                        # opens the MQTT session

    obs, info = env.reset()
    for _ in range(settings.robot.max_steps):
        action = policy(obs)             # consumer-supplied; carries h_t etc.
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from quaid_env.config import Settings
from quaid_env.mqtt_controller import MqttController
from quaid_env.observations import (
    OBSERVATION_SIZE,
    build_reset_observation,
    build_step_observation,
    yaw_delta_with_wrap,
)
from quaid_env.reward import RewardBreakdown, compute_reward
from quaid_env.sqlite_logger import SqliteLogger
from quaid_env.theta import ThetaController


log = logging.getLogger(__name__)


ACTION_SIZE = 8


class QuaidEnv(gym.Env):
    """``gymnasium.Env`` for the Quaid quadruped over MQTT."""

    metadata = {'render_modes': []}

    observation_space = spaces.Box(
        low=-2000.0, high=2000.0, shape=(OBSERVATION_SIZE,), dtype=np.float32,
    )
    action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32,
    )

    def __init__(
        self,
        settings: Settings,
        *,
        controller: Optional[MqttController] = None,
        theta_controller: Optional[ThetaController] = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.controller = controller or MqttController(settings)
        self.theta = theta_controller or ThetaController(settings.robot)

        self._steps = 0
        self._frame = 0
        self._episode_no = 0
        self._mean_yaw = 0.0
        self._last_yaw = 0.0
        self._last_position = (0.0, 0.0)
        self._last_step_ms = int(time.time() * 1000)
        # Deque of recent action vectors (np.float32, shape (8,)) — used by
        # the action-smoothness reward term, which inspects the last 3.
        self._action_history: deque[np.ndarray] = deque(maxlen=3)
        self._connected = False
        self._sqlite_logger: Optional[SqliteLogger] = None

    # --------------------------------------------------------------- setup
    def connect(self) -> None:
        """Open the MQTT session and push robot-side settings."""
        if self._connected:
            return
        self.controller.connect()
        self.controller.upload_settings()
        self.controller.start_streaming()
        self._connected = True

    def setup_logger(self, sqlite_path) -> None:
        """Open a per-run SQLite log file. Mirrors ``QuaidEnv::setup_logger``
        in the C++ env. Pass ``None`` (the default) to disable logging.

        The schema (4 tables: observations / actions / rewards /
        theta_updates) matches the C++ output 1:1 so dumps from either
        implementation can be loaded with the same tooling.
        """
        if sqlite_path is None:
            return
        self._sqlite_logger = SqliteLogger(sqlite_path)
        log.info('opened sqlite logger at %s', self._sqlite_logger.path)

    # ------------------------------------------------------------- reset --
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if not self._connected:
            self.connect()

        r = self.settings.robot

        # Mirror QuaidEnv.cpp:65-83: pick the reset action based on strategy,
        # then either run a theta adjust or send "pTH0" to reset the rotation.
        strategy = r.reset_strategy
        if strategy == 0:
            time.sleep(r.reset_wait / 1000.0)
        elif strategy == 1:
            self.controller.reset()
            time.sleep(r.reset_wait / 1000.0)
        else:
            self.controller.stand_up()

        if r.adjust_theta_on_reset and r.reset_theta_max_steps > 0:
            update, msg = self.theta.adjust(
                self.controller.data, self._mean_yaw, self._frame,
                self._episode_no, self._steps,
            )
            self.controller.message(msg)
            self._log_theta_update(update)
            time.sleep(0.2)
        else:
            self.controller.message('pTH0')

        self._episode_no += 1
        self._steps = 0
        self._action_history.clear()

        data = self.controller.data.snapshot()
        obs = build_reset_observation(data, self.settings.observations)

        self._last_yaw = data.yaw
        self._last_position = (data.position_x, data.position_y)
        self._last_step_ms = int(time.time() * 1000)

        info = {'episode': self._episode_no}
        return obs, info

    # -------------------------------------------------------------- step --
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_SIZE)
        self._steps += 1
        self._frame += 1
        self._action_history.append(action.copy())

        r = self.settings.robot
        terminated = False
        truncated = False

        # Send the action and wait one step period. If we've already hit
        # max_steps the C++ env skips take_step and just sleeps; we mirror
        # that so the env stays in lockstep with the robot.
        if self._steps >= r.max_steps or self.controller.data.snapshot().paused:
            truncated = self._steps >= r.max_steps
            terminated = self.controller.data.snapshot().paused
            time.sleep(r.step_time / 1000.0)
        else:
            self.controller.act(action)
            time.sleep(r.step_time / 1000.0)

        # Read fresh state and derive cross-step quantities.
        now_ms = int(time.time() * 1000)
        time_delta_centisec = (now_ms - self._last_step_ms) / 100.0
        self._last_step_ms = now_ms

        data = self.controller.data.snapshot()
        observation_age_centisec = (now_ms - data.received_time_ms) / 100.0

        if self.settings.reward.reward_type == 'euclidean':
            distance = math.hypot(data.position_x - self._last_position[0],
                                  data.position_y - self._last_position[1])
        else:
            distance = data.position_x - self._last_position[0]

        # Update mean yaw (the C++ formula caps the running average at
        # mean_yaw_steps samples — see QuaidEnv.cpp:181-186).
        denom = min(self._frame, r.mean_yaw_steps) + 1
        self._mean_yaw = (self._mean_yaw * (denom - 1) + abs(data.yaw)) / denom

        if r.yaw_end_episode != 0 and abs(data.yaw) > r.yaw_end_episode:
            log.info('yaw %.3f exceeded limit ±%.3f, terminating episode', data.yaw, r.yaw_end_episode)
            terminated = True

        yaw_delta = yaw_delta_with_wrap(self._last_yaw, data.yaw)

        obs = build_step_observation(
            data,
            self.settings.observations,
            distance=distance,
            yaw_delta=yaw_delta,
            time_delta_centisec=time_delta_centisec,
            observation_age_centisec=observation_age_centisec,
            reward_type=self.settings.reward.reward_type,
        )

        # The C++ env only computes the reward while still stepping (i.e. not
        # in the done-tail), see QuaidEnv.cpp:266-268.
        if self._steps < r.max_steps:
            breakdown = compute_reward(
                data=data,
                distance=distance,
                yaw_delta=yaw_delta,
                settings=self.settings.reward,
                action_history=list(self._action_history),
            )
            reward = breakdown.total
        else:
            breakdown = RewardBreakdown()
            reward = 0.0

        # Mid-episode theta adjust (QuaidEnv.cpp:273-281).
        if r.reset_theta_max_steps > 0 and self._frame >= self.theta.next_reset_frame and self._frame != 0:
            update, msg = self.theta.adjust(
                self.controller.data, self._mean_yaw, self._frame,
                self._episode_no, self._steps,
            )
            self.controller.message(msg)
            self._log_theta_update(update)
            self.theta.schedule_next(self._frame)
            time.sleep(0.2)
            data = self.controller.data.snapshot()
            self._last_position = (data.position_x, data.position_y)
            self._last_yaw = data.yaw
        else:
            self._last_position = (data.position_x, data.position_y)
            self._last_yaw = data.yaw

        # Per-step SQLite logging — same schema as QuaidLogging.cpp.
        done = terminated or truncated
        if self._sqlite_logger is not None:
            self._log_step(
                now_ms=now_ms, data=data, action=action, breakdown=breakdown,
                distance=distance, yaw_delta=yaw_delta,
                time_delta_centisec=time_delta_centisec,
                observation_age_centisec=observation_age_centisec,
                done=done,
            )
            if done:
                self._sqlite_logger.flush_episode()

        info = {
            'reward_breakdown': breakdown,
            'mean_yaw': self._mean_yaw,
            'episode': self._episode_no,
            'step': self._steps,
        }
        return obs, float(reward), terminated, truncated, info

    # ----------------------------------------------------- log helpers --
    def _log_step(self, *, now_ms, data, action, breakdown,
                  distance, yaw_delta, time_delta_centisec,
                  observation_age_centisec, done) -> None:
        """Buffer one observation / action / reward triple. Schema mirrors
        ``QuaidLogging.cpp``."""
        self._sqlite_logger.log_observation(
            episode_no=self._episode_no, step=self._steps,
            time=now_ms, time_delta=int(time_delta_centisec),
            distance=float(distance),
            voltage=float(data.voltage),
            current=float(data.current),
            yaw_delta=float(yaw_delta),
            yaw_mean=float(self._mean_yaw),
            yaw=float(data.yaw),
            pitch=float(data.pitch),
            roll=float(data.roll),
            obs_age=int(observation_age_centisec),
            # Servo order matches QuaidEnv.cpp:108-115 / 245-252:
            # BL knee, BL thigh, BR knee, BR thigh, FR knee, FR thigh,
            # FL knee, FL thigh.
            servo0=int(data.position_knee_back_left),
            servo1=int(data.position_thigh_back_left),
            servo2=int(data.position_knee_back_right),
            servo3=int(data.position_thigh_back_right),
            servo4=int(data.position_knee_front_right),
            servo5=int(data.position_thigh_front_right),
            servo6=int(data.position_knee_front_left),
            servo7=int(data.position_thigh_front_left),
            current_front_left=float(data.current_front_left),
            current_front_right=float(data.current_front_right),
            current_back_left=float(data.current_back_left),
            current_back_right=float(data.current_back_right),
            acc_x=float(data.acc_x), acc_y=float(data.acc_y), acc_z=float(data.acc_z),
            gyro_x=float(data.gyro_x), gyro_y=float(data.gyro_y), gyro_z=float(data.gyro_z),
            position_x=float(data.position_x),
            position_y=float(data.position_y),
            position_z=float(data.position_z),
            theta=float(data.theta),
            done=int(bool(done)),
        )
        self._sqlite_logger.log_action(
            episode_no=self._episode_no, step=self._steps,
            time=now_ms, action=action,
        )
        self._sqlite_logger.log_reward(
            episode_no=self._episode_no, step=self._steps, time=now_ms,
            reward=float(breakdown.total),
            speed=float(breakdown.speed),
            distance=float(breakdown.distance),
            yaw_delta=float(breakdown.yaw_delta),
            yaw=float(breakdown.yaw),
            pitch=float(breakdown.pitch),
            roll=float(breakdown.roll),
            z_position=float(breakdown.z),
            current_front_left=float(breakdown.current_front_left),
            current_front_right=float(breakdown.current_front_right),
            current_back_left=float(breakdown.current_back_left),
            current_back_right=float(breakdown.current_back_right),
            current=float(breakdown.current),
            acc_z=float(breakdown.acc_z),
            acc_x=float(breakdown.acc_x),
            acc_y=float(breakdown.acc_y),
            action_smoothness=float(breakdown.action_smoothness),
        )

    def _log_theta_update(self, update) -> None:
        if self._sqlite_logger is None:
            return
        self._sqlite_logger.log_theta_update(
            episode_no=self._episode_no, step=self._steps,
            time=int(time.time() * 1000),
            current_theta=float(update.old_theta),
            yaw=float(update.yaw),
            yaw_mean=float(update.mean_yaw),
            random=float(update.random_theta),
            new_theta=float(update.new_theta),
        )

    # --------------------------------------------------------------- close
    def close(self) -> None:
        if self._sqlite_logger is not None:
            self._sqlite_logger.close()
            self._sqlite_logger = None
        if self._connected:
            self.controller.disconnect()
            self._connected = False

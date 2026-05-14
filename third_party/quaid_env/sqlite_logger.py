"""Per-episode SQLite logger — Python port of ``QuaidLogging.cpp``.

Schema and field order match the C++ logger 1:1 so dumps from the Python
inference loop can be loaded by the same analysis tooling that consumes the
C++ outputs (e.g. ``data/snapshots/<env>/<policy>/<timestamp>/Quaid_*.sqlite``).

Four tables, all keyed by ``(episode_no, step)``:

* ``observations`` — full sensor + mocap state per step
* ``actions``      — 8-dim policy action per step
* ``rewards``      — reward breakdown per step (every term + total)
* ``theta_updates``— frame-rotation events (only when ``adjust_theta`` fires)

Rows are buffered per-episode and committed in a single transaction by
``flush_episode()``; this matches the C++ pattern in
``QuaidLogging.cpp:188`` and gives ~10x throughput vs autocommit.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


# SQL DDL for the four tables. Schema mirrors QuaidLogging.cpp:26-167 exactly.
_DDL = (
    """
    CREATE TABLE IF NOT EXISTS observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        episode_no INTEGER, step INTEGER, time INTEGER, time_delta INTEGER,
        distance REAL, voltage REAL, current REAL,
        yaw_delta REAL, yaw_mean REAL, yaw REAL, pitch REAL, roll REAL,
        obs_age INTEGER,
        servo0 INTEGER, servo1 INTEGER, servo2 INTEGER, servo3 INTEGER,
        servo4 INTEGER, servo5 INTEGER, servo6 INTEGER, servo7 INTEGER,
        current_front_left REAL, current_front_right REAL,
        current_back_left REAL, current_back_right REAL,
        acc_x REAL, acc_y REAL, acc_z REAL,
        gyro_x REAL, gyro_y REAL, gyro_z REAL,
        position_x REAL, position_y REAL, position_z REAL,
        theta REAL, done INTEGER
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        episode_no INTEGER, step INTEGER, time INTEGER,
        servo0 REAL, servo1 REAL, servo2 REAL, servo3 REAL,
        servo4 REAL, servo5 REAL, servo6 REAL, servo7 REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS rewards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        episode_no INTEGER, step INTEGER, time INTEGER,
        reward REAL, speed REAL, distance REAL,
        yaw_delta REAL, yaw REAL, pitch REAL, roll REAL, z_position REAL,
        current_front_left REAL, current_front_right REAL,
        current_back_left REAL, current_back_right REAL, current REAL,
        acc_z REAL, acc_x REAL, acc_y REAL, action_smoothness REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS theta_updates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        episode_no INTEGER, step INTEGER, time INTEGER,
        current_theta REAL, yaw REAL, yaw_mean REAL,
        random REAL, new_theta REAL
    );
    """,
)

# Column order for INSERTs. Locked here so future schema edits can be
# diffed against this single list.
_OBS_COLS = (
    'episode_no', 'step', 'time', 'time_delta',
    'distance', 'voltage', 'current',
    'yaw_delta', 'yaw_mean', 'yaw', 'pitch', 'roll', 'obs_age',
    'servo0', 'servo1', 'servo2', 'servo3',
    'servo4', 'servo5', 'servo6', 'servo7',
    'current_front_left', 'current_front_right',
    'current_back_left', 'current_back_right',
    'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z',
    'position_x', 'position_y', 'position_z', 'theta', 'done',
)
_ACTION_COLS = (
    'episode_no', 'step', 'time',
    'servo0', 'servo1', 'servo2', 'servo3',
    'servo4', 'servo5', 'servo6', 'servo7',
)
_REWARD_COLS = (
    'episode_no', 'step', 'time',
    'reward', 'speed', 'distance',
    'yaw_delta', 'yaw', 'pitch', 'roll', 'z_position',
    'current_front_left', 'current_front_right',
    'current_back_left', 'current_back_right', 'current',
    'acc_z', 'acc_x', 'acc_y', 'action_smoothness',
)
_THETA_COLS = (
    'episode_no', 'step', 'time',
    'current_theta', 'yaw', 'yaw_mean', 'random', 'new_theta',
)


def _build_insert(table: str, cols: tuple[str, ...]) -> str:
    qmarks = ', '.join('?' * len(cols))
    return f'INSERT INTO {table} ({", ".join(cols)}) VALUES ({qmarks})'


_INSERT_OBS = _build_insert('observations', _OBS_COLS)
_INSERT_ACTION = _build_insert('actions', _ACTION_COLS)
_INSERT_REWARD = _build_insert('rewards', _REWARD_COLS)
_INSERT_THETA = _build_insert('theta_updates', _THETA_COLS)


class SqliteLogger:
    """Buffered per-episode SQLite writer.

    Usage::

        logger = SqliteLogger('Quaid_2026-05-10T15-32-28.sqlite')
        # per step:
        logger.log_observation(episode_no=1, step=1, ...)
        logger.log_action(episode_no=1, step=1, time=ms, action=action_arr)
        logger.log_reward(episode_no=1, step=1, time=ms, breakdown=...)
        # at end of episode (terminated / truncated):
        logger.flush_episode()
        # at the end of the run:
        logger.close()
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._conn = sqlite3.connect(str(path))
        for stmt in _DDL:
            self._conn.execute(stmt)
        self._conn.commit()

        self._pending_obs: list[tuple] = []
        self._pending_actions: list[tuple] = []
        self._pending_rewards: list[tuple] = []
        self._pending_thetas: list[tuple] = []

    @property
    def path(self) -> Path:
        return self._path

    # --------------------------------------------------- observation ----
    def log_observation(self, **fields: Any) -> None:
        """Buffer one row for the ``observations`` table.

        Required keys mirror the C++ ``QuaidObsLog`` struct — see
        ``_OBS_COLS``. Pass them all by name; missing keys raise
        ``KeyError`` so a typo doesn't silently log NULLs.
        """
        self._pending_obs.append(tuple(fields[c] for c in _OBS_COLS))

    def log_action(self, *, episode_no: int, step: int, time: int, action) -> None:
        """Buffer one row for the ``actions`` table.

        ``action`` is the 8-dim policy output (NOT the 16-dim controller
        expansion). The C++ logger does the same.
        """
        if len(action) != 8:
            raise ValueError(f'expected 8-dim action, got {len(action)}')
        self._pending_actions.append(
            (episode_no, step, time, *(float(a) for a in action))
        )

    def log_reward(self, **fields: Any) -> None:
        """Buffer one row for the ``rewards`` table — every term + total."""
        self._pending_rewards.append(tuple(fields[c] for c in _REWARD_COLS))

    def log_theta_update(self, **fields: Any) -> None:
        """Buffer one row for the ``theta_updates`` table."""
        self._pending_thetas.append(tuple(fields[c] for c in _THETA_COLS))

    # ----------------------------------------------------- flush --------
    def flush_episode(self) -> None:
        """Write all buffered rows in a single transaction per table.

        Mirrors C++ ``log_observations`` / ``log_actions`` / ``log_reward`` /
        ``log_theta_updates``. Called by the env on episode-end.
        """
        with self._conn:
            if self._pending_obs:
                self._conn.executemany(_INSERT_OBS, self._pending_obs)
                self._pending_obs.clear()
            if self._pending_actions:
                self._conn.executemany(_INSERT_ACTION, self._pending_actions)
                self._pending_actions.clear()
            if self._pending_rewards:
                self._conn.executemany(_INSERT_REWARD, self._pending_rewards)
                self._pending_rewards.clear()
            if self._pending_thetas:
                self._conn.executemany(_INSERT_THETA, self._pending_thetas)
                self._pending_thetas.clear()

    def close(self) -> None:
        try:
            self.flush_episode()
        finally:
            self._conn.close()

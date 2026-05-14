"""Merged sensor + mocap state held by the controller.

This is the Python equivalent of the C++ ``QuaidData`` struct: the controller
mutates a single instance under a lock as MQTT packets arrive, and the env
reads ``snapshot()`` whenever it needs a coherent view (matching the C++
behaviour where ``shared_ptr<QuaidData>`` is held by both layers).
"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QuaidData:
    # Latest observation packet fields (see packets.ObservationPacket).
    time_delta: int = 0
    distance: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    voltage: float = 0.0
    current: float = 0.0

    position_knee_front_left: int = 0
    position_thigh_front_left: int = 0
    position_knee_front_right: int = 0
    position_thigh_front_right: int = 0
    position_knee_back_left: int = 0
    position_thigh_back_left: int = 0
    position_knee_back_right: int = 0
    position_thigh_back_right: int = 0

    current_front_left: float = 0.0
    current_front_right: float = 0.0
    current_back_left: float = 0.0
    current_back_right: float = 0.0

    acc_x: float = 0.0
    acc_y: float = 0.0
    acc_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0

    # Mocap-derived fields. The unit is **decimetres** (0.1 m), matching the
    # raw int16 the simulator publishes (MuJoCo metres * 10). The reward
    # YAML's ``target_distance_per_step`` and ``z_center`` are tuned to this
    # unit, so the controller keeps the raw value rather than converting to
    # metres — same as the C++ env. Real-hardware mocap publishes 2.5x
    # larger values; the controller divides by 2.5 in non-sim mode to match.
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0

    # Theta-rotated frame: theta is the current world-frame rotation,
    # `noRotationX/Y` and `noRotationYaw` are the raw mocap values before
    # rotation, used as pivots when theta is updated.
    theta: float = 0.0
    pivot_x: float = 0.0
    pivot_y: float = 0.0
    no_rotation_x: float = 0.0
    no_rotation_y: float = 0.0
    no_rotation_yaw: float = 0.0

    paused: bool = False
    received_time_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class SharedQuaidData:
    """Thread-safe holder around a ``QuaidData`` instance.

    Writers (the MQTT callback thread) call ``mutate(...)`` to apply
    incremental updates under a lock. Readers (the env thread) call
    ``snapshot()`` to get an immutable copy.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data = QuaidData()

    def snapshot(self) -> QuaidData:
        with self._lock:
            return copy.copy(self._data)

    def mutate(self, **updates) -> None:
        with self._lock:
            for k, v in updates.items():
                setattr(self._data, k, v)

    def with_lock(self):
        """Return the underlying lock for callers that need to apply
        multi-field updates atomically.

        Usage::

            with shared.with_lock() as data:
                data.position_x = ...
                data.position_y = ...
        """
        return _DataContext(self)


class _DataContext:
    def __init__(self, holder: SharedQuaidData) -> None:
        self._holder = holder

    def __enter__(self) -> QuaidData:
        self._holder._lock.acquire()
        return self._holder._data

    def __exit__(self, exc_type, exc, tb) -> None:
        self._holder._lock.release()

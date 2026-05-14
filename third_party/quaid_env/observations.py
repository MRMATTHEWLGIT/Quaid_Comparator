"""17-dim observation vector builder for the Quaid policy.

This is the most policy-coupled module in the package: trained policies
expect the EXACT field order produced here. See the slot table at the top of
``build_step_observation``. The C++ counterpart lives at
``sim-to-real-cpp/src/envs/quaid/QuaidEnv.cpp:135-264``.

Two builders are provided because the C++ env populates the obs vector
differently in ``reset()`` vs ``step()`` — the reset version skips fields
that depend on cross-step deltas (``distance``, ``yaw_delta``, ``time_delta``,
``observation_age``), the step version fills them in.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from quaid_env.config import ObservationSettings
from quaid_env.quaid_data import QuaidData


# ---- Slot indices (see QuaidEnv.h:57-74) -------------------------------------
IND_TIME_DELTA = 0
IND_DISTANCE = 1
IND_VOLTAGE = 2
IND_CURRENT = 3
IND_YAW = 4
IND_YAW_DELTA = 5
IND_PITCH = 6
IND_ROLL = 7
IND_OBS_AGE = 8
IND_SERVO_0 = 9

# Aliased slots — the C++ env reuses these positions for per-leg currents
# when the "primary" fields above are disabled.
IND_CURRENT_FRONT_LEFT = IND_TIME_DELTA       # 0
IND_CURRENT_FRONT_RIGHT = IND_DISTANCE        # 1
IND_CURRENT_BACK_LEFT = IND_VOLTAGE           # 2
IND_CURRENT_BACK_RIGHT = IND_OBS_AGE          # 8

# acc_z reuses the yaw_delta slot.
IND_ACC_Z = IND_YAW_DELTA                     # 5

OBSERVATION_SIZE = 17


# ---- Normalization -----------------------------------------------------------
# Constants from QuaidUtils.h:154-178.
MAX_CURRENT = 8.0
_NORM_MAX = {
    'x-distance': 2.0,
    'euclidean-distance': 2.0,
    'voltage': 2500.0,
    'current': MAX_CURRENT,
    'current_per_leg': MAX_CURRENT / 2.0,
    'yaw_delta': 1.0,
    'yaw': math.pi / 2,
    'pitch': math.pi / 2,
    'roll': math.pi / 2,
    'acc_z': 20.0,
    'servo0': 500.0,
    'servo1': 500.0,
    'servo2': 500.0,
    'servo3': 500.0,
    'servo4': 500.0,
    'servo5': 500.0,
    'servo6': 500.0,
    'servo7': 500.0,
}

# Stats for the tanh_estimator normaliser. From QuaidUtils.h:134-152, captured
# during a 2022-08-30 training run.
_TANH_STATS = {
    'x-distance': (-0.13644343891402716, 1.1011431936499758),
    'euclidean-distance': (-0.13644343891402716, 1.1011431936499758),
    'current': (0.19769225964705803, 2.0111728659726835),
    'yaw': (-0.23511277588853413, 0.35094668189349626),
    'yaw_delta': (-0.00033282582850679065, 0.0315493127782947),
    'pitch': (-0.01629855445229396, 0.06852792710789501),
    'roll': (-0.01629855445229396, 0.06852792710789501),
    'servo0': (261.9004615384615, 25.432159242208368),
    'servo1': (232.9882443438914, 24.075331125179986),
    'servo2': (330.94380995475115, 24.356104596046404),
    'servo3': (342.8862895927602, 29.304920441028532),
    'servo4': (316.13806334841627, 24.558300532180667),
    'servo5': (354.1294117647059, 30.635742878605356),
    'servo6': (252.3256108597285, 32.70852883491036),
    'servo7': (236.29462443438914, 30.873545234040872),
}


def normalize(field: str, value: float, settings: ObservationSettings) -> float:
    """Match ``QuaidEnv::normalize`` (QuaidEnv.cpp:437-448)."""
    if settings.normalization_type == 'tanh_estimator':
        mean, stddev = _TANH_STATS[field]
        return 0.5 * math.tanh(0.01 * (value - mean) / stddev)
    if settings.normalization_type == 'divide_by_max':
        return value / _NORM_MAX[field]
    if settings.normalization_type == 'quantize':
        return int((value / _NORM_MAX[field]) * settings.quantization_scale)
    raise ValueError(f'Unknown normalization_type: {settings.normalization_type!r}')


# ---- Servo wiring ------------------------------------------------------------
# Servo names map to QuaidData fields per QuaidEnv.cpp:245-252; the order is
# load-bearing because trained policies see this exact sequence.
_SERVO_FIELDS = (
    ('servo0', 'position_knee_back_left'),
    ('servo1', 'position_thigh_back_left'),
    ('servo2', 'position_knee_back_right'),
    ('servo3', 'position_thigh_back_right'),
    ('servo4', 'position_knee_front_right'),
    ('servo5', 'position_thigh_front_right'),
    ('servo6', 'position_knee_front_left'),
    ('servo7', 'position_thigh_front_left'),
)


def _write_servos(out: np.ndarray, data: QuaidData, settings: ObservationSettings) -> None:
    for i, (norm_key, attr) in enumerate(_SERVO_FIELDS):
        out[IND_SERVO_0 + i] = normalize(norm_key, getattr(data, attr), settings)


def _write_current_per_leg(out: np.ndarray, data: QuaidData, settings: ObservationSettings) -> None:
    """Overwrite slots 0/1/2/8 with the four per-leg currents.

    This is intentional — when ``time_delta`` / ``distance`` / ``voltage`` /
    ``observation_age`` are disabled in the YAML (the QuaidSIM-v4 case),
    those slots become the per-leg current observations. See
    QuaidEnv.cpp:255-260.
    """
    out[IND_CURRENT_FRONT_LEFT] = normalize('current_per_leg', data.current_front_left, settings)
    out[IND_CURRENT_FRONT_RIGHT] = normalize('current_per_leg', data.current_front_right, settings)
    out[IND_CURRENT_BACK_LEFT] = normalize('current_per_leg', data.current_back_left, settings)
    out[IND_CURRENT_BACK_RIGHT] = normalize('current_per_leg', data.current_back_right, settings)


def build_reset_observation(data: QuaidData, settings: ObservationSettings) -> np.ndarray:
    """Mirror ``QuaidEnv::reset`` (QuaidEnv.cpp:90-119).

    distance / time_delta / yaw_delta / observation_age are zeroed because no
    delta has accumulated yet. ``acc_z`` and per-leg currents still apply if
    enabled.
    """
    out = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

    out[IND_CURRENT_FRONT_LEFT] = 0.0
    out[IND_CURRENT_FRONT_RIGHT] = 0.0
    out[IND_CURRENT_BACK_LEFT] = 0.0
    out[IND_CURRENT_BACK_RIGHT] = 0.0

    out[IND_CURRENT] = 0.0  # not moving, no current
    if settings.yaw:
        out[IND_YAW] = normalize('yaw', data.yaw, settings)
    if settings.pitch:
        out[IND_PITCH] = normalize('pitch', data.pitch, settings)
    if settings.roll:
        out[IND_ROLL] = normalize('roll', data.roll, settings)
    out[IND_OBS_AGE] = 0.0

    _write_servos(out, data, settings)

    if settings.acc_z:
        out[IND_ACC_Z] = normalize('acc_z', data.acc_z, settings)

    return out


def build_step_observation(
    data: QuaidData,
    settings: ObservationSettings,
    *,
    distance: float,
    yaw_delta: float,
    time_delta_centisec: float,
    observation_age_centisec: float,
    reward_type: str,
) -> np.ndarray:
    """Mirror ``QuaidEnv::step`` observation construction (QuaidEnv.cpp:165-264).

    Slot layout (positions reused per the alias rules above):

      0  -> time_delta            OR current_front_left      (if current_per_leg)
      1  -> distance              OR current_front_right     (if current_per_leg)
      2  -> voltage               OR current_back_left       (if current_per_leg)
      3  -> current
      4  -> yaw
      5  -> yaw_delta             OR acc_z                   (if acc_z)
      6  -> pitch
      7  -> roll
      8  -> observation_age       OR current_back_right      (if current_per_leg)
      9..16 -> servo positions (BL knee, BL thigh, BR knee, BR thigh,
                                FR knee, FR thigh, FL knee, FL thigh)

    ``distance``, ``yaw_delta``, ``time_delta_centisec`` and
    ``observation_age_centisec`` come from the env (they need cross-step
    deltas the controller doesn't have).
    """
    out = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

    # Primary fields, gated by the YAML.
    if settings.time_delta:
        out[IND_TIME_DELTA] = time_delta_centisec
    if settings.distance:
        key = 'euclidean-distance' if reward_type == 'euclidean' else 'x-distance'
        out[IND_DISTANCE] = normalize(key, distance, settings)
    if settings.voltage:
        out[IND_VOLTAGE] = normalize('voltage', data.voltage, settings)
    if settings.current:
        out[IND_CURRENT] = normalize('current', data.current, settings)
    if settings.yaw:
        out[IND_YAW] = normalize('yaw', data.yaw, settings)
    if settings.yaw_delta:
        out[IND_YAW_DELTA] = normalize('yaw_delta', yaw_delta, settings)
    if settings.pitch:
        out[IND_PITCH] = normalize('pitch', data.pitch, settings)
    if settings.roll:
        out[IND_ROLL] = normalize('roll', data.roll, settings)
    if settings.observation_age:
        out[IND_OBS_AGE] = observation_age_centisec

    # Servos always.
    _write_servos(out, data, settings)

    # Aliased overrides — order matters: per-leg currents go AFTER the
    # primary fields so they overwrite slots 0/1/2/8 when enabled.
    if settings.current_per_leg:
        _write_current_per_leg(out, data, settings)
    if settings.acc_z:
        out[IND_ACC_Z] = normalize('acc_z', data.acc_z, settings)

    return out


def yaw_delta_with_wrap(last_yaw: float, current_yaw: float) -> float:
    """Reproduce the ``yawDelta`` calculation from QuaidEnv.cpp:193-199.

    The result is signed: positive when ``|current_yaw|`` decreased
    (returning toward zero), negative otherwise.
    """
    delta = abs(last_yaw - current_yaw)
    if delta > math.pi:
        delta = 2 * math.pi - delta
    if abs(current_yaw) > abs(last_yaw):
        delta = -delta
    return delta

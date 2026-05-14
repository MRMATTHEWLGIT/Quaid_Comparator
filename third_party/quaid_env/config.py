"""YAML config loader for the Quaid environment.

The C++ side uses OpenCV's ``%YAML:1.0`` flavour which is mostly standard
YAML with an extra ``%YAML:1.0`` directive line at the top. PyYAML chokes on
that line, so we strip it (and the directive end marker ``---``) before
parsing.

Every key from ``sim-to-real-cpp/src/envs/quaid/QuaidUtils.cpp`` is exposed
as a typed dataclass field. Defaults match the C++ ``Settings`` struct so a
sparse YAML file behaves identically to the C++ loader.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class PortsSettings:
    port: str = ''
    mqtt_server_ip: str = 'tcp://localhost:1883'
    mqtt_queue_no: str = '01'


@dataclass
class RobotSettings:
    sim: bool = True
    max_steps: int = 500
    step_time: int = 50               # ms per env step
    monitoring_delay: int = 50
    streaming_delay: int = 25
    acting_delay: int = 5
    reset_strategy: int = 0           # 0=sleep, 1=reset+sleep, 2=stand up
    reset_wait: int = 1000            # ms after reset
    exponential_filter: float = 0.75

    yaw_end_episode: float = 0.0      # 0 -> disabled

    min_x: float = 0.0
    max_x: float = 330.0
    min_y: float = 0.0
    max_y: float = 235.0
    debug_mocap: bool = False

    adjust_theta_on_reset: bool = False
    adjust_theta_interval: float = math.pi / 4
    adjust_theta_distribution: str = 'normal'                # zero | uniform | normal
    adjust_theta_alternate_distribution: str = 'normal'
    adjust_theta_alternate_with_forward_every: int = 0
    reset_theta_min_steps: int = 50
    reset_theta_max_steps: int = 500
    mean_yaw_steps: int = 10000

    offsets: dict = field(default_factory=dict)              # keyed by mqtt_queue_no
    sensor_noise: str = ''

    no_inference: bool = False
    env_logger: bool = True


@dataclass
class ObservationSettings:
    distance: bool = True
    observation_age: bool = False
    voltage: bool = True
    current: bool = True
    current_cut_off: float = 4.0
    current_per_leg: bool = True
    time_delta: bool = True
    yaw: bool = True
    yaw_delta: bool = True
    roll: bool = True
    pitch: bool = True
    acc_z: bool = True
    useIMUForYaw: bool = False

    normalization_type: str = 'divide_by_max'                # divide_by_max | tanh_estimator | quantize
    quantization_scale: float = 32767.5


@dataclass
class RewardSettings:
    reward_type: str = 'x-direction'                         # x-direction | euclidean
    distance_reward_ratio: float = 1.0
    speed_reward_ratio: float = 2.0
    target_distance_per_step: float = 1.0
    yaw_reward_ratio: float = -0.1
    yaw_delta_reward_ratio: float = 1.0
    pitch_reward_ratio: float = -0.1
    roll_reward_ratio: float = -0.1

    current_reward_ratio: float = -1.0
    current_per_leg_ratio: float = 0.0
    current_cut_off: float = 4.0

    z_reward_ratio: float = 0.0
    z_center: float = 2.0

    acc_z_reward_ratio: float = 1.0
    acc_x_reward_ratio: float = 1.0
    acc_y_reward_ratio: float = 1.0

    action_smoothness_reward_ratio: float = -0.01


@dataclass
class Settings:
    ports: PortsSettings = field(default_factory=PortsSettings)
    robot: RobotSettings = field(default_factory=RobotSettings)
    observations: ObservationSettings = field(default_factory=ObservationSettings)
    reward: RewardSettings = field(default_factory=RewardSettings)


def _strip_opencv_header(text: str) -> str:
    """Remove the ``%YAML:1.0`` directive line that OpenCV writes; PyYAML
    handles the ``---`` document marker on its own."""
    lines = text.splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('%YAML'):
            continue
        out.append(line)
    return '\n'.join(out)


def _populate(target: Any, data: Mapping[str, Any]) -> None:
    """Copy known keys from ``data`` into the dataclass ``target``,
    coercing types where necessary. Keys not present on the dataclass are
    silently ignored (forwards compatibility with newer YAMLs)."""
    if data is None:
        return
    known = {f.name: f.type for f in fields(target)}
    for key, value in data.items():
        if key not in known:
            continue
        ftype = known[key]
        # Coerce 0/1 -> bool when the field is annotated bool (OpenCV writes
        # ints for booleans). int/str don't need explicit coercion.
        if ftype is bool or ftype == 'bool':
            value = bool(int(value)) if isinstance(value, (int, float, str)) else bool(value)
        elif ftype is float or ftype == 'float':
            value = float(value)
        elif ftype is int or ftype == 'int':
            value = int(value)
        setattr(target, key, value)


def load(path: str | Path) -> Settings:
    """Load a Quaid YAML config and return a populated ``Settings`` tree."""
    text = Path(path).read_text()
    raw = yaml.safe_load(_strip_opencv_header(text)) or {}

    settings = Settings()
    _populate(settings.ports, raw.get('ports'))
    _populate(settings.robot, raw.get('robot'))
    _populate(settings.observations, raw.get('observations'))
    _populate(settings.reward, raw.get('reward'))

    # mqtt_queue_no is used in topic strings — coerce to str so int values
    # like `888` in the YAML produce `quaid/obs/r888BIN` rather than choking
    # on .upper() / concatenation later.
    settings.ports.mqtt_queue_no = str(settings.ports.mqtt_queue_no)

    # `current_cut_off` lives under `observations:` in the YAML but is also
    # consumed by the reward layer. Mirror it to keep call sites simple.
    settings.reward.current_cut_off = settings.observations.current_cut_off

    # `robot.offsets` is a nested mapping in the YAML, not a flat string.
    offsets = raw.get('robot', {}).get('offsets')
    if isinstance(offsets, Mapping):
        settings.robot.offsets = dict(offsets)

    return settings

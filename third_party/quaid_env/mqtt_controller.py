"""MQTT controller — Python port of ``QuaidControllerMqtt``.

Subscribes to the robot's observation, mocap, and ctrl topics; decodes the
binary packets and merges them into a ``SharedQuaidData`` instance under a
lock. Publishes actions and control messages back.

Threading: paho-mqtt's ``loop_start()`` runs the network/callback loop on a
background thread. Callbacks mutate ``SharedQuaidData`` under its internal
lock; the env calls ``snapshot()`` from the main thread.

Wire-level details are in ``QuaidControllerMqtt.cpp`` — most importantly,
the action payload is text in the format ``a<f>,<f>,...,<f>,0\\n`` (the
trailing ``,0\\n`` is intentional, matching the C++ ``ostream_iterator``
quirk at line 55-69).
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, Optional, Sequence

import paho.mqtt.client as mqtt

from quaid_env.config import Settings
from quaid_env.packets import (
    OBSERVATION_HEADER,
    MOCAP_HEADER,
    decode_mocap,
    decode_observation,
)
from quaid_env.quaid_data import SharedQuaidData


log = logging.getLogger(__name__)


# Last-will-and-testament payload published if the controller drops mid-run.
LWT_PAYLOAD = b'b\n'  # the robot interprets a leading 'b' as "stop"


class MqttController:
    """Owns the MQTT client and the live ``SharedQuaidData`` snapshot.

    The env constructs one instance per session, calls ``connect()`` /
    ``upload_settings()``, then drives ``act()`` / ``reset()`` / ``stand_up()``
    /``message()`` etc. as needed.
    """

    def __init__(self, settings: Settings, *, client: Optional[mqtt.Client] = None) -> None:
        self.settings = settings
        self.data = SharedQuaidData()

        q = settings.ports.mqtt_queue_no
        self._topic_obs = f'quaid/obs/r{q}BIN'
        self._topic_mocap = f'quaid/mocap/r{q}BIN'
        self._topic_ctrl = f'quaid/ctrl/r{q}'
        self._topic_act = f'quaid/act/r{q}'
        self._topic_set = f'quaid/set/r{q}'

        self._client = client or mqtt.Client()
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.will_set(self._topic_act, LWT_PAYLOAD, qos=1)

    # ------------------------------------------------------------------ life
    def connect(self) -> None:
        host, port = _parse_mqtt_uri(self.settings.ports.mqtt_server_ip)
        log.info('connecting to mqtt broker %s:%d', host, port)
        self._client.connect(host, port, keepalive=60)
        self._client.loop_start()

    def disconnect(self) -> None:
        try:
            self._client.loop_stop()
        finally:
            self._client.disconnect()

    # -------------------------------------------------------- subscriptions
    def _on_connect(self, client: mqtt.Client, userdata, flags, rc) -> None:  # noqa: D401
        if rc != 0:
            log.error('mqtt connect failed with rc=%s', rc)
            return
        log.info('mqtt connected (rc=0)')
        client.subscribe([(self._topic_obs, 1), (self._topic_mocap, 1), (self._topic_ctrl, 1)])

    def _on_message(self, client: mqtt.Client, userdata, msg) -> None:
        payload = msg.payload
        if not payload:
            return
        header = payload[0]
        try:
            if header == OBSERVATION_HEADER and msg.topic == self._topic_obs:
                self._handle_observation(payload)
            elif header == MOCAP_HEADER and msg.topic == self._topic_mocap:
                self._handle_mocap(payload)
            elif msg.topic == self._topic_ctrl:
                self._handle_ctrl(payload)
            else:
                log.debug('unhandled message on topic=%s header=0x%02X', msg.topic, header)
        except Exception:  # don't let a malformed packet kill the callback thread
            log.exception('failed to handle message on topic=%s', msg.topic)

    def _handle_observation(self, payload: bytes) -> None:
        pkt = decode_observation(payload)
        self.data.mutate(
            time_delta=pkt.time_delta,
            distance=pkt.distance,
            yaw=pkt.yaw,
            pitch=pkt.pitch,
            roll=pkt.roll,
            voltage=pkt.voltage,
            current=pkt.current,
            position_knee_front_left=pkt.position_knee_front_left,
            position_thigh_front_left=pkt.position_thigh_front_left,
            position_knee_front_right=pkt.position_knee_front_right,
            position_thigh_front_right=pkt.position_thigh_front_right,
            position_knee_back_left=pkt.position_knee_back_left,
            position_thigh_back_left=pkt.position_thigh_back_left,
            position_knee_back_right=pkt.position_knee_back_right,
            position_thigh_back_right=pkt.position_thigh_back_right,
            current_front_left=pkt.current_front_left,
            current_front_right=pkt.current_front_right,
            current_back_left=pkt.current_back_left,
            current_back_right=pkt.current_back_right,
            acc_x=pkt.acc_x,
            acc_y=pkt.acc_y,
            acc_z=pkt.acc_z,
            gyro_x=pkt.gyro_x,
            gyro_y=pkt.gyro_y,
            gyro_z=pkt.gyro_z,
            received_time_ms=int(time.time() * 1000),
        )

    def _handle_mocap(self, payload: bytes) -> None:
        pkt = decode_mocap(payload)
        # The simulator publishes int16 positions in **decimetres** (0.1 m) —
        # quaid-sim-cpp/src/mqtt_controller.cpp:235 casts ``sensordata * 10``
        # to int16, where ``sensordata`` is MuJoCo's mocap site in metres. The
        # YAML's ``target_distance_per_step`` and ``z_center`` are tuned for
        # this dm unit, so we keep the raw value rather than converting to
        # metres. Real-hardware mocap publishes 2.5x larger values that need
        # /2.5 to match the simulator's scale — same as
        # QuaidControllerMqtt.cpp:217-222.
        scale = 1.0 if self.settings.robot.sim else 2.5
        x = pkt.x / scale
        y = pkt.y / scale
        z = pkt.z / scale

        with self.data.with_lock() as data:
            data.position_x = x
            data.position_y = y
            data.position_z = z
            data.no_rotation_x = x
            data.no_rotation_y = y
            data.no_rotation_yaw = pkt.yaw
            # The "rotated" yaw the policy observes is mocap yaw minus the
            # current frame rotation (theta is initially 0).
            data.yaw = pkt.yaw - data.theta
            data.pitch = pkt.pitch
            data.roll = pkt.roll

    def _handle_ctrl(self, payload: bytes) -> None:
        text = payload.decode(errors='replace').strip()
        if text.startswith('P'):
            try:
                self.data.mutate(paused=bool(int(text[1:])))
            except ValueError:
                log.warning('malformed pause ctrl: %r', text)
        # 'R' (reload settings) and 'O<deg>' (mocap offset) are robot-side
        # operations and don't need state changes here.

    # ---------------------------------------------------- send to the robot
    def _send(self, topic: str, payload: bytes) -> None:
        self._client.publish(topic, payload, qos=1)

    def act(self, action: Sequence[float]) -> None:
        """Publish an 8-dim policy action.

        The C++ side expands 8 actions to 16 servo positions (4 legs x [knee,
        thigh, 0, 0]); we mirror that so the on-MCU parser sees the same
        wire format. See ``QuaidEnv::take_step`` (QuaidEnv.cpp:362-374).

        Action wire format: ``a<f>,<f>,...,<f>,0\\n`` — the trailing ``,0``
        is intentional (matches the C++ ``ostream_iterator`` quirk).
        """
        if len(action) != 8:
            raise ValueError(f'expected 8-dim action, got {len(action)}')
        servos = [
            action[0], action[1], 0, 0,
            action[2], action[3], 0, 0,
            action[4], action[5], 0, 0,
            action[6], action[7], 0, 0,
        ]
        body = ','.join(_fmt(v) for v in servos)
        self._send(self._topic_act, f'a{body},0\n'.encode())

    def reset(self) -> None:
        """Publish ``s\\n`` and sleep 2 s, mirroring
        ``QuaidControllerMqtt::reset`` (QuaidControllerMqtt.cpp:121-126)."""
        self._send(self._topic_act, b's\n')
        time.sleep(2.0)

    def sit_down(self) -> None:
        self._send(self._topic_act, b'r\n')
        time.sleep(2.0)

    def stand_up(self) -> None:
        self._send(self._topic_act, b'e\n')
        time.sleep(2.0)

    def start_streaming(self) -> None:
        self._send(self._topic_act, b'x\n')

    def stop_streaming(self) -> None:
        self._send(self._topic_act, b'y\n')

    def request_observation(self) -> None:
        self._send(self._topic_act, b'z\n')

    def message(self, msg: str) -> None:
        """Publish a control message to BOTH ctrl and set topics, mirroring
        ``QuaidControllerMqtt::message`` (QuaidControllerMqtt.cpp:114-119)."""
        payload = (msg + '\n').encode()
        self._send(self._topic_ctrl, payload)
        self._send(self._topic_set, payload)

    # --------------------------------------------------- settings sync ----
    def upload_settings(self) -> None:
        """Push robot-side timing / filter / offset / noise / current
        threshold settings, matching ``QuaidEnv::upload_settings`` ->
        ``QuaidControllerMqtt::uploadSettings``.
        """
        r = self.settings.robot
        o = self.settings.observations
        q = self.settings.ports.mqtt_queue_no

        self._send(self._topic_act, f'u{r.streaming_delay}\n'.encode())
        self._send(self._topic_act, f'i{r.acting_delay}\n'.encode())
        self._send(self._topic_act, f'f{r.exponential_filter}\n'.encode())
        self._send(self._topic_act, f'c{int(o.current_cut_off)}\n'.encode())

        offset = r.offsets.get(f'r{q}') if isinstance(r.offsets, dict) else None
        if offset:
            self._send(self._topic_act, f'o{offset}\n'.encode())
        if r.sensor_noise:
            self._send(self._topic_act, f'n{r.sensor_noise}\n'.encode())


def _fmt(v: float) -> str:
    """Match the C++ ``std::ostream_iterator<float>`` default formatting,
    which uses ``%g``-style with default precision."""
    return f'{float(v):g}'


def _parse_mqtt_uri(uri: str) -> tuple[str, int]:
    """Accept ``tcp://host:port`` or ``host:port`` or just ``host``."""
    s = uri
    if '://' in s:
        s = s.split('://', 1)[1]
    if ':' in s:
        host, port_s = s.rsplit(':', 1)
        return host, int(port_s)
    return s, 1883

"""Wire-format codecs for the binary packets the robot publishes over MQTT.

Layouts mirror the C++ structs in
``sim-to-real-cpp/include/envs/quaid/QuaidStateObservations.h`` and
``QuaidMocapData.h`` — both are ``__attribute__((packed))`` on x86/ARM, so
the Python ``struct`` format strings use ``=`` for native byte order with no
padding.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


# ---- Observation packet (header byte 0x0A) -----------------------------------
#
# struct QuaidStateObservations {
#     uint8_t  header;                                       // 0x0A
#     int16_t  time_delta;
#     float    distance, yaw, pitch, roll;
#     float    voltage, current;
#     int16_t  position_knee_front_left,  position_thigh_front_left;
#     int16_t  position_knee_front_right, position_thigh_front_right;
#     int16_t  position_knee_back_left,   position_thigh_back_left;
#     int16_t  position_knee_back_right,  position_thigh_back_right;
#     float    current_front_left, current_front_right,
#              current_back_left,  current_back_right;
#     float    acc_x,  acc_y,  acc_z;
#     float    gyro_x, gyro_y, gyro_z;
# } __attribute__((packed));
#
# Total size: 1 + 2 + 6*4 + 8*2 + 4*4 + 6*4 = 83 bytes.
OBSERVATION_HEADER = 0x0A
_OBS_FMT = '=B h ffff ff hhhhhhhh ffff fff fff'.replace(' ', '')
OBS_STRUCT = struct.Struct(_OBS_FMT)
assert OBS_STRUCT.size == 83, f'observation struct size mismatch: {OBS_STRUCT.size}'


@dataclass(frozen=True)
class ObservationPacket:
    header: int
    time_delta: int
    distance: float
    yaw: float
    pitch: float
    roll: float
    voltage: float
    current: float
    position_knee_front_left: int
    position_thigh_front_left: int
    position_knee_front_right: int
    position_thigh_front_right: int
    position_knee_back_left: int
    position_thigh_back_left: int
    position_knee_back_right: int
    position_thigh_back_right: int
    current_front_left: float
    current_front_right: float
    current_back_left: float
    current_back_right: float
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


def decode_observation(payload: bytes) -> ObservationPacket:
    fields = OBS_STRUCT.unpack(payload)
    pkt = ObservationPacket(*fields)
    if pkt.header != OBSERVATION_HEADER:
        raise ValueError(f'observation header mismatch: 0x{pkt.header:02X}')
    return pkt


def encode_observation(pkt: ObservationPacket) -> bytes:
    """Build a packet from a dataclass — only used by tests / smoke harness."""
    return OBS_STRUCT.pack(*(getattr(pkt, f) for f in ObservationPacket.__dataclass_fields__))


# ---- Mocap packet (header byte 0x0E) -----------------------------------------
#
# typedef struct {
#     uint8_t  header;        // 0x0E
#     bool     degrees;
#     uint8_t  rigid_body_no;
#     int16_t  x, y, z;       // mm
#     float    yaw, pitch, roll;
#     float    qr, qi, qj, qk;
# } __attribute__((packed)) QuaidMocapData;
#
# Total: 1 + 1 + 1 + 3*2 + 7*4 = 37 bytes.
MOCAP_HEADER = 0x0E
_MOCAP_FMT = '=B?Bhhhfffffff'
MOCAP_STRUCT = struct.Struct(_MOCAP_FMT)
assert MOCAP_STRUCT.size == 37, f'mocap struct size mismatch: {MOCAP_STRUCT.size}'


@dataclass(frozen=True)
class MocapPacket:
    header: int
    degrees: bool
    rigid_body_no: int
    x: int
    y: int
    z: int
    yaw: float
    pitch: float
    roll: float
    qr: float
    qi: float
    qj: float
    qk: float


def decode_mocap(payload: bytes) -> MocapPacket:
    fields = MOCAP_STRUCT.unpack(payload)
    pkt = MocapPacket(*fields)
    if pkt.header != MOCAP_HEADER:
        raise ValueError(f'mocap header mismatch: 0x{pkt.header:02X}')
    return pkt


def encode_mocap(pkt: MocapPacket) -> bytes:
    return MOCAP_STRUCT.pack(*(getattr(pkt, f) for f in MocapPacket.__dataclass_fields__))

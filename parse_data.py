from __future__ import annotations
import struct
from dataclasses import dataclass

# =============================================================================
# Global Constants
# =============================================================================

# The binary format of the observation data
OBS_BIN_FORMAT = "<Bhffffffhhhhhhhhffffffffff"

# The binary format of the mocap data
MOCAP_BIN_FORMAT = "<B?Bhhhfffffff"

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Observation:
    """A class to store the most recent observation data."""
    timestamp: str
    header: int
    time_delta: int
    distance: int
    yaw: int
    pitch: int
    roll: int
    voltage: int
    current: int
    position_knee_front_left: int
    position_thigh_front_left: int
    position_knee_front_right: int
    position_thigh_front_right: int
    position_knee_back_left: int
    position_thigh_back_left: int
    position_knee_back_right: int
    position_thigh_back_right: int
    current_front_left: int
    current_front_right: int
    current_back_left: int
    current_back_right: int
    acc_x: int
    acc_y: int
    acc_z: int
    gyro_x: int
    gyro_y: int
    gyro_z: int


@dataclass
class MocapObservation:
    """A class to store the most recent mocap data."""
    timestamp: str
    header: int
    degrees: int
    rigid_body_no: int
    x: int
    y: int
    z: int
    yaw: int
    pitch: int
    roll: int
    qr: int
    qi: int
    qj: int
    qk: int

# =============================================================================
# MQTT Callback Functions
# =============================================================================

def parse_obs_data(topic, payload, timestamp):
    """Parse the observation data from the MQTT message payload."""

    # An invalid payload length has been detected
    if len(payload) != struct.calcsize(OBS_BIN_FORMAT):
        print(f"Invalid payload length for {topic}: {len(payload)}")
        return None

    # Unpack the observation data
    unpacked_data = struct.unpack(OBS_BIN_FORMAT, payload)

    # Create and return the observation data
    return Observation(
        timestamp=timestamp,
        header=unpacked_data[0],
        time_delta=unpacked_data[1],
        distance=unpacked_data[2],
        yaw=unpacked_data[3],
        pitch=unpacked_data[4],
        roll=unpacked_data[5],
        voltage=unpacked_data[6],
        current=unpacked_data[7],
        position_knee_front_left=unpacked_data[8],
        position_thigh_front_left=unpacked_data[9],
        position_knee_front_right=unpacked_data[10],
        position_thigh_front_right=unpacked_data[11],
        position_knee_back_left=unpacked_data[12],
        position_thigh_back_left=unpacked_data[13],
        position_knee_back_right=unpacked_data[14],
        position_thigh_back_right=unpacked_data[15],
        current_front_left=unpacked_data[16],
        current_front_right=unpacked_data[17],
        current_back_left=unpacked_data[18],
        current_back_right=unpacked_data[19],
        acc_x=unpacked_data[20],
        acc_y=unpacked_data[21],
        acc_z=unpacked_data[22],
        gyro_x=unpacked_data[23],
        gyro_y=unpacked_data[24],
        gyro_z=unpacked_data[25],
    )


def parse_mocap_data(topic, payload, timestamp):
    """Parse the mocap data from the MQTT message payload."""

    # An invalid payload length has been detected
    if len(payload) != struct.calcsize(MOCAP_BIN_FORMAT):
        print(f"Invalid payload length for {topic}: {len(payload)}")
        return None

    # Unpack the mocap data
    unpacked_data = struct.unpack(MOCAP_BIN_FORMAT, payload)

    # Create and return the mocap data
    return MocapObservation(
        timestamp=timestamp,
        header=unpacked_data[0],
        degrees=unpacked_data[1],
        rigid_body_no=unpacked_data[2],
        x=unpacked_data[3],
        y=unpacked_data[4],
        z=unpacked_data[5],
        yaw=unpacked_data[6],
        pitch=unpacked_data[7],
        roll=unpacked_data[8],
        qr=unpacked_data[9],
        qi=unpacked_data[10],
        qj=unpacked_data[11],
        qk=unpacked_data[12],
    )
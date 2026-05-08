from __future__ import annotations

import signal
import sys
from datetime import datetime

import paho.mqtt.client as mqtt
import struct

# =============================================================================
# Global Constants
# =============================================================================

# The MQTT host and port
MQTT_HOST = "127.0.0.1"
MQTT_PORT = 1883
MQTT_KEEPALIVE = 60

# What queue number is the data coming from?
Q_NUMBER = 200

# The main topics to listen to
OBS_BIN_TOPIC = f"quaid/obs/r{Q_NUMBER}BIN"
MOCAP_BIN_TOPIC = f"quaid/mocap/r{Q_NUMBER}BIN"

# The client ID to use for the MQTT connection
CLIENT_ID = "quaid_binary_listener"

OBS_BIN_FORMAT = "<Bhffffffhhhhhhhhffffffffff"
MOCAP_BIN_FORMAT = "<B?Bhhhfffffff"

# =============================================================================
# MQTT Callback Functions
# =============================================================================

def on_connect(client, userdata, flags, reason_code, properties=None):
    """Called when the client connects to the MQTT broker."""

    # The connection was successful
    if reason_code == 0:
        print(f"Connected to MQTT broker on {MQTT_HOST}:{MQTT_PORT}")

        # Subscrive to the main topics
        client.subscribe(OBS_BIN_TOPIC)
        client.subscribe(MOCAP_BIN_TOPIC)

        print(f"Subscribed to {OBS_BIN_TOPIC} and {MOCAP_BIN_TOPIC}")

    # The connection was unsuccessful
    else:
        print(f"Failed to connect to MQTT broker: {reason_code}")


def on_message(client, userdata, message):
    """Called when a message is received from the MQTT broker."""

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    topic = message.topic
    payload = message.payload

    if topic == OBS_BIN_TOPIC:

        if len(payload) != struct.calcsize(OBS_BIN_FORMAT):
            print(f"Invalid payload length for {topic}: {len(payload)}")
            return

        observations = struct.unpack(OBS_BIN_FORMAT, payload)

        header = observations[0]
        time_delta = observations[1]
        distance = observations[2]
        yaw = observations[3]
        pitch = observations[4]
        roll = observations[5]
        voltage = observations[6]
        current = observations[7]
        position_knee_front_left = observations[8]
        position_thigh_front_left = observations[9]
        position_knee_front_right = observations[10]
        position_thigh_front_right = observations[11]
        position_knee_back_left = observations[12]
        position_thigh_back_left = observations[13]
        position_knee_back_right = observations[14]
        position_thigh_back_right = observations[15]
        current_front_left = observations[16]
        current_front_right = observations[17]
        current_back_left = observations[18]
        current_back_right = observations[19]
        acc_x = observations[20]
        acc_y = observations[21]
        acc_z = observations[22]
        gyro_x = observations[23]
        gyro_y = observations[24]
        gyro_z = observations[25]

        print(f"Header: {header}")
        print(f"Time delta: {time_delta}")
        print(f"Distance: {distance}")
        print(f"Yaw: {yaw}")
        print(f"Pitch: {pitch}")
        print(f"Roll: {roll}")
        print(f"Voltage: {voltage}")
        print(f"Current: {current}")
        print(f"Position knee front left: {position_knee_front_left}")
        print(f"Position thigh front left: {position_thigh_front_left}")
        print(f"Position knee front right: {position_knee_front_right}")
        print(f"Position thigh front right: {position_thigh_front_right}")
        print(f"Position knee back left: {position_knee_back_left}")
        print(f"Position thigh back left: {position_thigh_back_left}")
        print(f"Position knee back right: {position_knee_back_right}")
        print(f"Position thigh back right: {position_thigh_back_right}")
        print(f"Current front left: {current_front_left}")
        print(f"Current front right: {current_front_right}")
        print(f"Current back left: {current_back_left}")
        print(f"Current back right: {current_back_right}")
        print(f"Acc x: {acc_x}")
        print(f"Acc y: {acc_y}")
        print(f"Acc z: {acc_z}")
        print(f"Gyro x: {gyro_x}")
        print(f"Gyro y: {gyro_y}")
        print(f"Gyro z: {gyro_z}")

    if topic == MOCAP_BIN_TOPIC:
        if len(payload) != struct.calcsize(MOCAP_BIN_FORMAT):
            print(f"Invalid payload length for {topic}: {len(payload)}")
            return

        mocap_observations = struct.unpack(MOCAP_BIN_FORMAT, payload)

        header = mocap_observations[0]
        degrees = mocap_observations[1]
        rigid_body_no = mocap_observations[2]
        x = mocap_observations[3]
        y = mocap_observations[4]
        z = mocap_observations[5]
        yaw = mocap_observations[6]
        pitch = mocap_observations[7]
        roll = mocap_observations[8]
        qr = mocap_observations[9]
        qi = mocap_observations[10]
        qj = mocap_observations[11]
        qk = mocap_observations[12]

        print(f"Header: {header}")
        print(f"Degrees: {degrees}")
        print(f"Rigid body no: {rigid_body_no}")
        print(f"X: {x}")
        print(f"Y: {y}")
        print(f"Z: {z}")
        print(f"Yaw: {yaw}")
        print(f"Pitch: {pitch}")
        print(f"Roll: {roll}")
        print(f"Qr: {qr}")
        print(f"Qi: {qi}")
        print(f"Qj: {qj}")
        print(f"Qk: {qk}")


def on_disconnect(client, userdata, reason_code):
    """Called when the client disconnects from the MQTT broker."""
    print(f"Disconnected from MQTT broker: {reason_code}")


def shutdown_handler(signum, frame):
    """Called when the program is interrupted with Ctrl+C."""
    print("\nShutting down...")
    sys.exit(0)


# =============================================================================
# Main
# =============================================================================

def main():
    """
    Connect to MQTT and listen forever.
    """

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    client = mqtt.Client(
        client_id=CLIENT_ID,
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    print(f"[INFO] Connecting to MQTT broker at {MQTT_HOST}:{MQTT_PORT}...")

    client.connect(
        host=MQTT_HOST,
        port=MQTT_PORT,
        keepalive=MQTT_KEEPALIVE,
    )

    client.loop_forever()


if __name__ == "__main__":
    main()
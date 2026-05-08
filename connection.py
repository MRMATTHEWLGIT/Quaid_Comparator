from __future__ import annotations

import signal
import sys
from datetime import datetime

import paho.mqtt.client as mqtt
import parse_data as pd

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

    # An observation data message has been received
    if topic == OBS_BIN_TOPIC:

        # Parse the observation data
        obs_data = pd.parse_obs_data(topic, payload, timestamp)

        # A valid observation data message has been received so add it to the queue
        if obs_data is not None:
            userdata["obs_queue"].put(obs_data)

    # A mocap data message has been received
    if topic == MOCAP_BIN_TOPIC:

        # Parse the mocap data
        mocap_data = pd.parse_mocap_data(topic, payload, timestamp)

        # A valid mocap data message has been received so add it to the queue
        if mocap_data is not None:
            userdata["mocap_queue"].put(mocap_data) 


def on_disconnect(client, userdata, reason_code):
    """Called when the client disconnects from the MQTT broker."""
    print(f"Disconnected from MQTT broker: {reason_code}")


def shutdown_handler(signum, frame):
    """Called when the program is interrupted with Ctrl+C."""
    print("\nShutting down...")
    sys.exit(0)


def start_mqtt_connection(obs_queue, mocap_queue):
    """Start the MQTT connection and return the client."""

    # Set up the signal handlers for Ctrl+C and SIGTERM
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Create the MQTT client
    client = mqtt.Client(
        client_id=CLIENT_ID,
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        userdata={
            "obs_queue": obs_queue,
            "mocap_queue": mocap_queue,
        },
    )

    # Add the callback functions to the client
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    print(f"[INFO] Connecting to MQTT broker at {MQTT_HOST}:{MQTT_PORT}...")

    # Connect to the MQTT broker
    client.connect(
        host=MQTT_HOST,
        port=MQTT_PORT,
        keepalive=MQTT_KEEPALIVE,
    )

    # Start the MQTT loop
    client.loop_start()

    return client
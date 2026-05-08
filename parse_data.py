from __future__ import annotations

import signal
import sys
from datetime import datetime

import paho.mqtt.client as mqtt

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

CLIENT_ID = "quaid_binary_listener"

PRINT_HEX_BYTES = 80

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

    hex_preview = payload[:PRINT_HEX_BYTES].hex(" ")

    print("=" * 80)
    print(f"Time       : {timestamp}")
    print(f"Topic      : {topic}")
    print(f"Bytes      : {len(payload)}")
    print(f"Hex preview: {hex_preview}")

    if len(payload) > PRINT_HEX_BYTES:
        print(f"... truncated, showing first {PRINT_HEX_BYTES} bytes")

    print("=" * 80)
    print()


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
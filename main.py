from __future__ import annotations
from queue import Queue, Empty
import connection as conn
import parse_data as pd

# =============================================================================
# Global Constants
# =============================================================================

# The timeout for the observation and mocap queues
QUEUE_TIMEOUT = 0.1

# =============================================================================
#  Main Loop
# =============================================================================

def main():
    """
    Connect to MQTT and listen forever.
    """

    # Stores the observations and mocap data as they arrive
    obs_queue = Queue()
    mocap_queue = Queue()

    # Start the MQTT connection
    client = conn.start_mqtt_connection(obs_queue, mocap_queue)

    # Stores the latest observation data and mocap data
    obs = None
    latest_mocap = None

    try: 

        while True:

            # Get the next observation from the queue
            try: 
                # Get the next observation from the queue
                obs = obs_queue.get(timeout=QUEUE_TIMEOUT)
            except Empty:
                pass

            # Get the most recent mocap data to match with the observation
            try:
                while True:
                    latest_mocap = mocap_queue.get_nowait()
            except Empty:
                pass

            if latest_mocap is None:
                print("Got obs, but no mocap data yet")
                continue

            print(obs)
            print(latest_mocap)


    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        client.loop_stop()
        client.disconnect()
        print("MQTT connection closed")


if __name__ == "__main__":
    main()
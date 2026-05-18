"""Thread-safe MQTT telemetry state for the live Streamlit dashboard."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import paho.mqtt.client as mqtt


log = logging.getLogger(__name__)


@dataclass
class LiveTelemetrySnapshot:
    """Immutable-ish snapshot returned to the Streamlit render loop."""

    connected: bool
    last_received_time: float | None
    current_episode_no: int | None
    last_message: dict[str, Any] | None
    steps: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


class LiveTelemetryState:
    """
    MQTT subscriber that stores only the currently active episode.

    The comparator publishes one JSON message per timestep. Whenever the episode
    number changes, this state clears the stored step list so the dashboard plots
    reset automatically per episode.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        topic: str,
        max_steps: int = 1000,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.topic = topic
        self.max_steps = int(max_steps)

        self._lock = threading.Lock()
        self._connected = False
        self._last_received_time: float | None = None
        self._current_episode_no: int | None = None
        self._last_message: dict[str, Any] | None = None
        self._steps: deque[dict[str, Any]] = deque(maxlen=self.max_steps)
        self._events: deque[dict[str, Any]] = deque(maxlen=200)

        self._client = mqtt.Client()
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        self._client.connect(self.host, self.port, keepalive=60)
        self._client.loop_start()

    def _on_connect(self, client: mqtt.Client, userdata, flags, rc) -> None:
        with self._lock:
            self._connected = rc == 0

        if rc == 0:
            client.subscribe(self.topic, qos=0)
            log.info("Dashboard subscribed to MQTT topic %s", self.topic)
        else:
            log.error("Dashboard MQTT connection failed with rc=%s", rc)

    def _on_disconnect(self, client: mqtt.Client, userdata, rc) -> None:
        with self._lock:
            self._connected = False

    def _on_message(self, client: mqtt.Client, userdata, message) -> None:
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except Exception:
            log.exception("Failed to decode dashboard telemetry payload")
            return

        if not isinstance(payload, dict):
            return

        self._store_payload(payload)

    def _store_payload(self, payload: dict[str, Any]) -> None:
        event_type = str(payload.get("event_type", "step"))
        episode_no = _optional_int(payload.get("episode_no"))

        with self._lock:
            # A new episode explicitly resets the dashboard state.
            if event_type == "episode_start":
                self._current_episode_no = episode_no
                self._steps.clear()
                self._events.clear()
                self._events.append(payload)

            # If a step arrives from a different episode, reset automatically.
            elif episode_no is not None and episode_no != self._current_episode_no:
                self._current_episode_no = episode_no
                self._steps.clear()
                self._events.clear()

            if event_type == "step":
                self._steps.append(payload)

                if int(payload.get("switch_committed", 0) or 0) == 1:
                    self._events.append(payload)

            elif event_type != "episode_start":
                self._events.append(payload)

            self._last_message = payload
            self._last_received_time = time.time()

    def snapshot(self) -> LiveTelemetrySnapshot:
        with self._lock:
            return LiveTelemetrySnapshot(
                connected=self._connected,
                last_received_time=self._last_received_time,
                current_episode_no=self._current_episode_no,
                last_message=dict(self._last_message) if self._last_message else None,
                steps=list(self._steps),
                events=list(self._events),
            )

    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None

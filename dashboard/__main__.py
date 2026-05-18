"""Run the comparator dashboard package with `python -m dashboard`."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """
    Launch the Streamlit app while preserving package-style arguments.

    Example:
        python -m dashboard \
            --comparator-database Models/Comparator/comparator_database.npz \
            --mqtt-topic quaid/comparator/r11/telemetry
    """

    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is not installed. Install it with: pip install streamlit plotly paho-mqtt"
        ) from exc

    app_path = Path(__file__).with_name("app.py")

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--",
        *sys.argv[1:],
    ]

    return int(streamlit_cli.main())


if __name__ == "__main__":
    raise SystemExit(main())

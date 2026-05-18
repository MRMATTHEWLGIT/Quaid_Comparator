"""
Main entrypoint for policy inference and comparator-based policy switching
within the Quaid environment.

This module supports:
    - Single-policy inference
    - Comparator-driven multi-policy switching
    - ONNX and TorchScript actor execution
    - MQTT-based communication with QuaidSim / sim-to-real pipelines
    - Online embedding generation for comparator selection

Comparator mode loads multiple locomotion policies simultaneously and
dynamically selects between them during execution based on comparator
logic operating within the learned embedding space.

Portions of the inference infrastructure were adapted from:

https://github.com/real-world-drl/esp-dl-quant-icra2026

Original project licensed under the MIT License.
Original authors retain copyright for their respective contributions.

Modifications and extensions in this repository include:
    - Comparator-based policy switching
    - Embedding-space policy selection
    - Multi-policy runtime management
    - Online GRU embedding inference
    - UMAP-based policy comparison and selection
"""

from __future__ import annotations

import argparse
import logging
import shutil
import yaml
import time
from pathlib import Path
from third_party.quaid_env import QuaidEnv, load_settings
from player import ComparatorPlayer

# Default max steps for the Quaid environment
DEFAULT_MAX_STEPS = 500


def get_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run policy inference and comparator evaluation in a Quaid environment.",
    )

    parser.add_argument(
        "--comparator-config",
        default="config/comparator.yaml",
        help="YAML file describing all policies used by comparator mode."
    )

    parser.add_argument(
        "--initial-policy",
        default=None,
        help="Initial policy used before the comparator makes a selection. "
             "Can also be specified in the policy config YAML."
    )

    parser.add_argument(
        "-c", "--env-config",
        required=True,
        help="YAML config for QuaidEnv."
    )

    parser.add_argument(
        "-q", "--mqtt-queue",
        help="Override mqtt_queue_no from the YAML."
    )

    parser.add_argument(
        "-g", "--gru-path",
        default=None,
        help="Optional global GRU override path. "
            "When provided, this replaces the GRU paths specified in the "
            "comparator policy YAML configuration."
    )

    parser.add_argument("-s", "--max-steps", type=int, default=DEFAULT_MAX_STEPS)

    parser.add_argument(
        "--step-delay-ms",
        type=int,
        default=0,
        help="Optional extra sleep between steps."
    )

    parser.add_argument("--policy-rnn-layers", type=int, default=3)
    parser.add_argument("--policy-rnn-hidden-size", type=int, default=64)

    parser.add_argument("--embedding-gru-layers", type=int, default=1)
    parser.add_argument("--embedding-gru-hidden-size", type=int, default=64)

    parser.add_argument(
        "--output-root",
        default="data",
        help="Parent directory for the per-run timestamped folder."
    )

    parser.add_argument(
        "--env-name",
        default=None,
        help="Used as the second-level folder name."
    )

    parser.add_argument(
        "--policy-name",
        default=None,
        help="Used as the third-level folder name."
    )

    parser.add_argument(
        "--no-logger",
        action="store_true",
        help="Disable per-step SQLite logging."
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for TorchScript paths."
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true"
    )

    parser.add_argument(
        "--dashboard-mqtt",
        action="store_true",
        help="Publish live comparator timestep telemetry for the Streamlit dashboard.",
    )

    parser.add_argument(
        "--dashboard-mqtt-topic",
        default=None,
        help="MQTT topic for dashboard telemetry. Defaults to "
            "quaid/comparator/r<queue>/telemetry.",
    )

    args = parser.parse_args(argv)

    # Validate the comparator config
    if args.comparator_config is None:
        raise ValueError("--comparator-config is required.")

    return args


def make_playbook_acronym(comparator_config: dict) -> str:
    """
    Create a short acronym for the episode playbook.

    Example:
        [flat, ramp, uneven, comparator, comparator]
        -> FRUCC
    """

    episode_playbook = comparator_config.get("episode_playbook", [])

    acronym_map = {
        "flat": "F",
        "ramp": "R",
        "uneven": "U",
        "comparator": "C",
    }

    acronym_parts = []

    for entry in episode_playbook:
        entry = str(entry)

        if entry in acronym_map:
            acronym_parts.append(acronym_map[entry])

        else:
            # Fallback for unexpected policy names.
            acronym_parts.append(entry[:1].upper())

    if len(acronym_parts) == 0:
        return "NO_PLAYBOOK"

    return "".join(acronym_parts)


def create_run_dir(
    args: argparse.Namespace,
    comparator_config: dict,
    timestamp: str | None = None,
) -> Path:
    """
    Create and return the timestamped output directory for the current run.

    Runs are saved directly one level under the configured output root:
        data/<timestamp>_<playbook_acronym>/
    """

    playbook_acronym = make_playbook_acronym(comparator_config)

    run_dir = Path(args.output_root) / f"{timestamp}_{playbook_acronym}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run output: {run_dir}")

    return run_dir


def copy_run_configs(*, comparator_config_path: str | Path | None, env_config_path: str | Path,
                     output_dir: Path) -> None:
    """
    Copy the environment and comparator configs into the run output directory.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_config_path = Path(env_config_path)

    if not env_config_path.exists():
        raise FileNotFoundError(f"Environment config does not exist: {env_config_path}")

    shutil.copy2(
        env_config_path,
        output_dir / "env.yaml",
    )

    if comparator_config_path is None:
        return

    comparator_config_path = Path(comparator_config_path)

    if not comparator_config_path.exists():
        raise FileNotFoundError(
            f"Comparator config does not exist: {comparator_config_path}"
        )

    shutil.copy2(
        comparator_config_path,
        output_dir / "comparator.yaml",
    )


def main(argv=None) -> int:

    # Parse command-line arguments
    args = get_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s',
    )

    # Create the timestamp for the run
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")

    # Load the comparator configuration
    with open(args.comparator_config, "r", encoding="utf-8") as file:
        comparator_config = yaml.safe_load(file)

    # Create the run directory
    run_dir = create_run_dir(
        args=args,
        comparator_config=comparator_config,
        timestamp=timestamp,
    )

    # The comparator config defines both policies and the episode playbook
    comparator_config_path = Path(args.comparator_config)

    copy_run_configs(
        comparator_config_path=comparator_config_path,
        env_config_path=args.env_config,
        output_dir=run_dir,
    )

    # Load the environment settings (.yaml file)
    env_settings = load_settings(args.env_config)

    if args.mqtt_queue is not None:

        # Override the MQTT queue specified in the environment YAML config.
        # Stored as a string to match the QuaidEnv configuration format.
        env_settings.ports.mqtt_queue_no = str(args.mqtt_queue)

    # Establish the MQTT topic for the dashboard telemetry
    dashboard_mqtt_topic = args.dashboard_mqtt_topic
    if dashboard_mqtt_topic is None:
        mqtt_queue_no = env_settings.ports.mqtt_queue_no
        dashboard_mqtt_topic = f"quaid/comparator/r{mqtt_queue_no}/telemetry"

    if args.dashboard_mqtt:
        logging.info("Dashboard MQTT telemetry topic: %s", dashboard_mqtt_topic)

    # Create the environment and connect to it
    env = QuaidEnv(env_settings)
    env.connect()

    # Setup the SQLite logger if requested
    if not args.no_logger:
        env.setup_logger(run_dir / f"Quaid_{timestamp}.sqlite")

    # Override the max steps if requested
    if (args.max_steps != DEFAULT_MAX_STEPS):
        env.settings.robot.max_steps = args.max_steps


    # Create the comparator player
    comparator_player = ComparatorPlayer(
        env=env,
        comparator_config=comparator_config,
        output_dir=run_dir,
        max_test_steps=args.max_steps,
        test_step_delay_ms=args.step_delay_ms,
        policy_rnn_layers=args.policy_rnn_layers,
        policy_rnn_hidden_size=args.policy_rnn_hidden_size,
        embedding_gru_layers=args.embedding_gru_layers,
        embedding_gru_hidden_size=args.embedding_gru_hidden_size,
        dashboard_mqtt_enabled=args.dashboard_mqtt,
        dashboard_mqtt_topic=dashboard_mqtt_topic,
    )

    # Run the comparator player
    try:
        comparator_player.play()
        comparator_player.stats.print_summary()

    except KeyboardInterrupt:
        logging.info("Comparator run interrupted by user.")
        return 130

    except Exception:
        logging.exception("Comparator run failed.")
        return 1

    finally:
        comparator_player.close()
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
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
import re
import sys
import yaml
import time
from pathlib import Path
from quaid_env import QuaidEnv, load_settings
from player import ComparatorPlayer


def get_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run policy inference and comparator evaluation in a Quaid environment.",
    )

    parser.add_argument(
        "--mode",
        choices=["single", "comparator"],
        default="single",
        help="Run either one fixed policy or the comparator policy-switching system."
    )

    parser.add_argument(
        "-m", "--model",
        help="Path to the actor model for single-policy inference (.onnx / .dat)."
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

    parser.add_argument("-e", "--episodes", type=int, default=5)

    parser.add_argument("-s", "--max-steps", type=int, default=500)

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
        default="data/snapshots",
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

    args = parser.parse_args(argv)

    if args.mode == "single":

        if args.model is None:
            parser.error("--model is required when --mode single is used.")

        if args.comparator_config is not None:
            parser.error(
                "--policy-config should only be used with --mode comparator."
            )

    elif args.mode == "comparator":

        if args.comparator_config is None:
            parser.error(
                "--policy-config is required when --mode comparator is used."
            )

        if args.model is not None:
            parser.error(
                "--model should only be used with --mode single."
            )

    return args


def detect_model_info(model_path: str) -> tuple[str, str]:
    """
    Infer the environment and policy names from an actor model filename.

    Example:
        ''aug_act_net_QuaidSIM-Flat_RA-TD3_+337.452.onnx''

    Returns:
        ('QuaidSIM-Flat', 'Flat')
    """

    name = Path(model_path).name
    match = re.search(r"act_net_(QuaidSIM-([^-_]+))_", name)

    if not match:
        return "unknown", "unknown"

    env_name = match.group(1)
    policy_name = match.group(2)

    return env_name, policy_name


def create_run_dir(args: argparse.Namespace, timestamp: str | None = None) -> Path:
    """
    Create and return the timestamped output directory for the current run.
    """

    # Single-policy mode
    if args.mode == "single":

        detected_env_name, detected_policy_name = detect_model_info(args.model)

        # Override the environment and policy names if provided
        env_name = args.env_name or detected_env_name
        policy_name = args.policy_name or detected_policy_name

    # Comparator mode
    elif args.mode == "comparator":

        # Override the environment and policy names if provided
        env_name = args.env_name or "QuaidSIM"
        policy_name = args.policy_name or "comparator"

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    # Create the run directory
    run_dir = Path(args.output_root) / env_name / policy_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run output: {run_dir}")

    return run_dir


def main(argv=None) -> int:

    # Parse command-line arguments
    args = get_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s',
    )

    # Create the timestamp for the run
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")

    # Create the run directory
    run_dir = create_run_dir(args, timestamp)

    # Load the environment settings (.yaml file)
    env_settings = load_settings(args.env_config)

    if args.mqtt_queue is not None:

        # Override the MQTT queue specified in the environment YAML config.
        # Stored as a string to match the QuaidEnv configuration format.
        env_settings.ports.mqtt_queue_no = str(args.mqtt_queue)

    # Create the environment and connect to it
    env = QuaidEnv(env_settings)
    env.connect()

    # Setup the SQLite logger if requested
    if not args.no_logger:
        env.setup_logger(run_dir / f"Quaid_{timestamp}.sqlite")


    comparator_config = None

    # Comparator mode
    if args.mode == "comparator":

        # Load the comparator configuration
        with open(args.comparator_config, "r", encoding="utf-8") as file:
            comparator_config = yaml.safe_load(file)

        # Create the comparator player
        comparator_player = ComparatorPlayer(
            env=env,
            comparator_config=comparator_config,
            output_dir=run_dir,
            test_episodes=args.episodes,
            max_test_steps=args.max_steps,
            test_step_delay_ms=args.step_delay_ms,
            policy_rnn_layers=args.policy_rnn_layers,
            policy_rnn_hidden_size=args.policy_rnn_hidden_size,
            embedding_gru_layers=args.embedding_gru_layers,
            embedding_gru_hidden_size=args.embedding_gru_hidden_size,
        )


    # Single policy mode
    elif args.mode == "single":
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
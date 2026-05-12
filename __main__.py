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
import time
from pathlib import Path


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
        "--policy-config",
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

    parser.add_argument("--rnn-layers", type=int, default=3)

    parser.add_argument("--rnn-hidden-size", type=int, default=64)

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

        if args.policy_config is not None:
            parser.error(
                "--policy-config should only be used with --mode comparator."
            )

    elif args.mode == "comparator":

        if args.policy_config is None:
            parser.error(
                "--policy-config is required when --mode comparator is used."
            )

        if args.model is not None:
            parser.error(
                "--model should only be used with --mode single."
            )

    return args
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from runner import OnnxRnnPolicyRunner
from stats import InferenceStats
from preprocessor import AddActionsPreprocessor
from comparator import Comparator

log = logging.getLogger(__name__)


class ComparatorPlayer:
    """
    Run comparator-based multi-policy inference in a Quaid environment.

    The player loads all actor policies specified in the comparator config,
    uses the comparator to select a policy at each timestep, executes the
    selected policy, and records per-step rollout statistics.
    """

    def __init__(
        self,
        env,
        comparator_config: dict,
        *,
        policy_rnn_layers: int = 3,
        policy_rnn_hidden_size: int = 64,
        embedding_gru_layers: int = 3,
        embedding_gru_hidden_size: int = 64,
        output_dir: Optional[str] = None,
        test_episodes: int = 5,
        max_test_steps: int = 500,
        test_step_delay_ms: int = 0,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.comparator_config = comparator_config

        # Store the RNN layers and hidden size
        self.policy_rnn_layers = policy_rnn_layers
        self.policy_rnn_hidden_size = policy_rnn_hidden_size

        # Store the embedding GRU layers and hidden size
        self.embedding_gru_layers = embedding_gru_layers
        self.embedding_gru_hidden_size = embedding_gru_hidden_size

        # Store the run variables
        self.output_dir = output_dir
        self.test_episodes = test_episodes
        self.max_test_steps = max_test_steps
        self.test_step_delay_ms = test_step_delay_ms
        self.device = device

        # Get the action and observation dimensions
        self._action_dim = int(np.prod(env.action_space.shape))
        self._obs_dim = int(np.prod(env.observation_space.shape))

        # Load the comparator configurations
        self.initial_policy = comparator_config["initial_policy"]
        self.policy_configs = comparator_config["policies"]
        self.comparator_assets_dir = comparator_config["comparator_assets_dir"]

        # Build the policy runners
        self.policy_runners = self._build_policy_runners()

        # Build the comparator
        self.comparator = self._build_comparator()

        # Build the preprocessor to add the previous action to the observation
        self.preprocessor = AddActionsPreprocessor(action_dim=self._action_dim)

        self.stats = InferenceStats(output_dir=output_dir)


    def _build_policy_runners(self) -> dict[str, OnnxRnnPolicyRunner]:
        """
        Build one stateless ONNX recurrent policy runner for each configured policy.
        """

        if not self.policy_configs:
            raise ValueError("Comparator config does not define any policies.")

        policy_runners = {}

        for policy_name, policy_config in self.policy_configs.items():

            if "model_path" not in policy_config:
                raise KeyError(f"Policy '{policy_name}' is missing required key 'model_path'.")

            # Extract the model path from the policy configuration
            model_path = policy_config["model_path"]

            # Build the policy runner based on the model path
            policy_runners[policy_name] = OnnxRnnPolicyRunner(
                model_path=model_path,
                rnn_layers=self.policy_rnn_layers,
                rnn_hidden_size=self.policy_rnn_hidden_size,
            )

        if self.initial_policy not in policy_runners:
            raise ValueError(
                f"Initial policy '{self.initial_policy}' is not defined in comparator policies."
            )

        return policy_runners

    
    def _build_comparator(self) -> Comparator:
        """
        Build the runtime comparator from the configured asset directory.
        """
        return Comparator(
            comparator_assets_dir=self.comparator_assets_dir,
            providers=["CPUExecutionProvider"],
        )
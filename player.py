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

        # Load the initial policy and policy configurations
        self.initial_policy = comparator_config["initial_policy"]
        self.policy_configs = comparator_config["policies"]

        # Build the policy runners
        self.policy_runners = self._build_policy_runners()

        # Build the comparator
        self.comparator = self._build_comparator()

        # Build the preprocessor to add the previous action to the observation
        self.preprocessor = AddActionsPreprocessor(action_dim=self._action_dim)

        self.stats = InferenceStats(output_dir=output_dir)
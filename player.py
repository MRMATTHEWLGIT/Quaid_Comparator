from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional
from collections import deque

import numpy as np

from runner import OnnxRnnPolicyRunner
from stats import InferenceStats, EpisodeStats
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
        self.comparator_hyperparameters = comparator_config["hyperparameters"]

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

        # Build the set of valid policy keys
        valid_policy_keys=set(self.policy_configs.keys())

        return Comparator(
            comparator_assets_dir=self.comparator_assets_dir,
            valid_policy_keys=valid_policy_keys,
            comparator_hyperparameters=self.comparator_hyperparameters,
            providers=["CPUExecutionProvider"],
        )


    def play(self) -> InferenceStats:
        """
        Run comparator-based policy-switching rollouts.
        """

        # Ensure the environment is reset before the loop
        obs, obs_raw, _ = self.env.reset()
        time.sleep(1.0)

        for episode_no in range(self.test_episodes):

            # Reset the environment after each episode
            obs, obs_raw, _ = self.env.reset()

            # Reset the comparator after each episode
            self.comparator.reset()

            # Store the policy history for the comparator
            policy_history = deque(maxlen=self.comparator.sequence_length)

            # Reset the previous action to zero
            prev_action = np.zeros(self._action_dim, dtype=np.float32)

            # Shared recurrent hidden state across all policy runners
            h_t = np.zeros(
                (self.policy_rnn_layers, 1, self.policy_rnn_hidden_size),
                dtype=np.float32,
            )

            # Start from the configured initial policy
            current_policy = self.initial_policy

            # Track when the last accepted policy switch occurred
            last_switch_step = -10_000

            episode_reward = 0.0
            episode_start = time.perf_counter()
            inference_times: list[int] = []

            self._wait_if_paused()

            for step in range(self.max_test_steps):

                # Save the current policy to the policy history
                policy_history.append(current_policy)

                inference_start = time.perf_counter_ns()

                if current_policy not in self.policy_runners:
                    raise KeyError(f"Unknown current policy: {current_policy}")

                # ------------------------------------------------------------------
                # Policy inference block
                # ------------------------------------------------------------------

                policy_state = self.preprocessor.process(obs, prev_action)

                runner = self.policy_runners[current_policy]
                action, h_t = runner.select_action(policy_state, h_t)

                action = np.clip(
                    np.asarray(action, dtype=np.float32),
                    self.env.action_space.low,
                    self.env.action_space.high,
                )

                # ------------------------------------------------------------------
                # Environment step block
                # ------------------------------------------------------------------

                obs, obs_raw, reward, terminated, truncated, _info = self.env.step(action)

                episode_reward += float(reward)

                # ------------------------------------------------------------------
                # Comparator policy-selection block
                # ------------------------------------------------------------------

                # Comparator uses the post-step raw observation paired with action_t.
                # This matches the SQLite logging/training data:
                #     env.step(action_t) -> obs_raw_{t+1}
                #     logged row        -> obs_raw_{t+1} + action_t
                comparator_state = np.concatenate(
                    [
                        np.asarray(obs_raw, dtype=np.float32),
                        np.asarray(action, dtype=np.float32),
                    ],
                    axis=0,
                )

                # feature_names = [
                #     "current_front_left", "current_front_right", "current_back_left",
                #     "current", "yaw", "acc_z", "pitch", "roll", "current_back_right",
                #     "servo0", "servo1", "servo2", "servo3",
                #     "servo4", "servo5", "servo6", "servo7",
                #     "action0", "action1", "action2", "action3",
                #     "action4", "action5", "action6", "action7",
                # ]

                # print("\nLIVE COMPARATOR STATE")
                # for name, value in zip(feature_names, comparator_state):
                #     print(f"{name:>20}: {value: .6f}")

                self.comparator.update_query_history(comparator_state)

                next_policy = self.comparator.select_policy(
                    current_policy=current_policy,
                    step=step,
                )

                if next_policy not in self.policy_runners:
                    raise KeyError(f"Comparator selected unknown policy: {next_policy}")

                inference_times.append((time.perf_counter_ns() - inference_start) // 1000)

                # ------------------------------------------------------------------
                # Policy switch commit block
                # ------------------------------------------------------------------

                # Determine if the policy history is full and if it is pure
                history_is_full = len(policy_history) == self.comparator.sequence_length
                history_is_policy_pure = all(
                    policy == current_policy
                    for policy in policy_history
                )

                # Extract the last step info from the comparator
                step_info = self.comparator.last_step_info

                switch_committed = False
                can_switch = (history_is_full and history_is_policy_pure)

                # Store the can_switch flag in the step info
                if step_info is not None:
                    step_info.can_switch = can_switch

                if next_policy != current_policy and can_switch:

                    previous_policy = current_policy
                    current_policy = next_policy
                    last_switch_step = step
                    switch_committed = True

                    log.info(
                        "Comparator switched policy at episode=%d step=%d: %s -> %s",
                        episode_no + 1,
                        step,
                        previous_policy,
                        current_policy,
                    )

                elif next_policy != current_policy:

                    log.debug(
                        "Comparator switch blocked by cooldown at episode=%d step=%d: %s -> %s",
                        episode_no + 1,
                        step,
                        current_policy,
                        next_policy,
                    )

                # Record the comparator step information if it exists
                if step_info is not None:
                    step_info.switch_committed = switch_committed

                    self.stats.record_comparator_step(
                        episode_no=episode_no,
                        step_info=step_info,
                    )

                # ------------------------------------------------------------------
                # Previous-action update block
                # ------------------------------------------------------------------

                prev_action = action.copy()

                if terminated or truncated:
                    break

                if self.test_step_delay_ms > 0:
                    time.sleep(self.test_step_delay_ms / 1000.0)

            else:
                step = self.max_test_steps - 1

            wall = time.perf_counter() - episode_start

            ep = EpisodeStats(
                episode_no=episode_no,
                reward=episode_reward,
                steps=step + 1,
                wall_seconds=wall,
                inference_times_us=inference_times,
            )

            self.stats.record_episode(ep)

            log.info(
                "episode %d: reward=%.3f steps=%d fps=%.2f mean_inf_us=%.1f",
                episode_no + 1,
                ep.reward,
                ep.steps,
                ep.fps,
                ep.mean_inference_us,
            )

            # Pause between episodes so the robot can be reset physically
            if episode_no < self.test_episodes - 1:
                self._pause_between_episodes()

            self._wait_if_paused()

        return self.stats
    

    def _pause_between_episodes(self) -> None:
        """
        Pause the robot between episodes until the controller receives P0.
        """

        try:
            log.info("Pausing robot between episodes with P1.")
            self.env.controller.message("P1")

        except AttributeError:
            log.warning("Environment controller does not support message('P1').")
            return

        # Give the controller time to publish and update paused state
        time.sleep(0.5)

        # Wait until the controller is unpaused again, usually after P0 is sent
        self._wait_if_paused()


    def _wait_if_paused(self) -> None:
        """
        Block while the environment reports a paused state.

        QuaidEnv exposes:
            env.controller.data.snapshot().paused

        Custom environments may instead expose:
            env.paused
        """

        while True:

            paused = False

            # QuaidEnv pause path
            try:
                paused = bool(self.env.controller.data.snapshot().paused)

            # Fallback for simpler/custom environments
            except AttributeError:
                paused = bool(getattr(self.env, "paused", False))

            if not paused:
                return

            log.info("Environment paused — waiting...")

            time.sleep(2.0)


    def close(self) -> None:
        """Close all runtime resources."""

        for policy_name, runner in self.policy_runners.items():
            log.info("Closing policy runner: %s", policy_name)
            runner.close()

        self.comparator.close()
        self.stats.close()
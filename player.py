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
        test_episodes: int | None = None,
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

        # Load the episode playbook. The playbook controls both the episode count and
        # whether each episode is run as a fixed-policy baseline or comparator episode.
        self.episode_playbook = self._load_episode_playbook()

        # The number of episodes is determined by the playbook length
        self.test_episodes = len(self.episode_playbook)

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


    def _load_episode_playbook(self) -> list[str]:
        """
        Load and validate the episode playbook from comparator_config.

        Each playbook entry must either be:
            - "comparator", meaning run the comparator normally from initial_policy
            - a policy key, meaning run that fixed policy for the whole episode
        """

        # No episode playbook was provided
        if "episode_playbook" not in self.comparator_config:
            raise KeyError(
                "Comparator config is missing required key 'episode_playbook'. "
                "Example: episode_playbook: [flat, ramp, uneven, comparator]"
            )

        # Extract the episode playbook from the comparator config
        episode_playbook = self.comparator_config["episode_playbook"]

        # Validate the episode playbook
        if not isinstance(episode_playbook, list):
            raise TypeError("episode_playbook must be a list.")
        if len(episode_playbook) == 0:
            raise ValueError("episode_playbook must contain at least one episode entry.")

        # Build the set of valid policy keys and the comparator key
        valid_entries = set(self.policy_configs.keys())
        valid_entries.add("comparator")

        cleaned_playbook = []

        for index, entry in enumerate(episode_playbook):

            entry = str(entry)

            # Not a valid policy key or comparator key
            if entry not in valid_entries:
                raise ValueError(
                    f"Invalid episode_playbook entry at index {index}: '{entry}'. "
                    f"Expected one of: {sorted(valid_entries)}"
                )

            # Add the valid entry to the cleaned playbook
            cleaned_playbook.append(entry)

        # Return the cleaned playbook
        return cleaned_playbook

    
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

        for episode_no, playbook_entry in enumerate(self.episode_playbook):

            # Reset the environment after each episode
            obs, obs_raw, _ = self.env.reset()

            # Reset the comparator after each episode
            self.comparator.reset()

            # Determine whether this episode should use the comparator or a fixed policy
            episode_uses_comparator = playbook_entry == "comparator"

            # Store the policy history for the comparator
            policy_history = deque(maxlen=self.comparator.sequence_length)

            # Reset the previous action to zero
            prev_action = np.zeros(self._action_dim, dtype=np.float32)

            # Shared recurrent hidden state across all policy runners
            h_t = np.zeros(
                (self.policy_rnn_layers, 1, self.policy_rnn_hidden_size),
                dtype=np.float32,
            )

            # Start either from the configured initial policy for comparator episodes,
            # or from the fixed policy requested by the playbook.
            if episode_uses_comparator:
                current_policy = self.initial_policy
            else:
                current_policy = playbook_entry

            log.info(
                "Starting episode %d/%d using playbook entry '%s' with initial policy '%s'.",
                episode_no + 1,
                self.test_episodes,
                playbook_entry,
                current_policy,
            )

            episode_start = time.perf_counter()
            inference_times: list[int] = []

            # Track episode reward breakdown
            reward_env = 0.0
            episode_performance_reward = 0.0

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

                reward_env += float(reward)

                # Extract the reward breakdown from the environment info
                reward_breakdown = _info.get("reward_breakdown", None)

                # Calculate the performance reward from the reward breakdown
                if reward_breakdown is not None:
                    reward_distance = float(reward_breakdown.distance)
                    reward_roll = float(reward_breakdown.roll)
                    reward_current = float(reward_breakdown.current)
                    reward_yaw = float(reward_breakdown.yaw)
                    reward_pitch = float(reward_breakdown.pitch)
                    reward_action_smoothness = float(reward_breakdown.action_smoothness)

                    step_performance_reward = (
                        reward_distance
                        + reward_roll
                        + reward_current
                        + reward_yaw
                        + reward_pitch
                        + reward_action_smoothness
                    )

                else:
                    reward_distance = None
                    reward_roll = None
                    reward_current = None
                    reward_yaw = None
                    reward_pitch = None
                    reward_action_smoothness = None

                    # Fallback only. Normally reward_breakdown should exist
                    step_performance_reward = float(reward)

                episode_performance_reward += float(step_performance_reward)

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

                self.comparator.update_query_history(comparator_state)

                # Run the comparator every step so that fixed-policy baseline episodes
                # still produce query embeddings, vote counts, UMAP coordinates, and
                # reward logs. For fixed-policy episodes, the comparator decision is
                # recorded for analysis but is not allowed to change the active policy.
                proposed_policy = self.comparator.select_policy(
                    current_policy=current_policy,
                    step=step,
                )

                if proposed_policy not in self.policy_runners:
                    raise KeyError(f"Comparator selected unknown policy: {proposed_policy}")

                # Comparator episodes can use the comparator's proposed policy.
                # Fixed-policy episodes ignore the proposal and keeps the playbook policy.
                if episode_uses_comparator:
                    next_policy = proposed_policy
                else:
                    next_policy = current_policy

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
                can_switch = (episode_uses_comparator and history_is_full and history_is_policy_pure)

                # Store the can_switch flag in the step info
                if step_info is not None:
                    step_info.can_switch = can_switch

                if next_policy != current_policy and can_switch:

                    previous_policy = current_policy
                    current_policy = next_policy
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
                        "Comparator switch blocked at episode=%d step=%d: %s -> %s "
                        "(episode_uses_comparator=%s)",
                        episode_no + 1,
                        step,
                        current_policy,
                        next_policy,
                        episode_uses_comparator,
                    )

                # Record the comparator step information if it exists
                if step_info is not None:
                    step_info.switch_committed = switch_committed

                    # For fixed-policy episodes, record the next policy as the current policy
                    if not episode_uses_comparator:
                        step_info.next_policy = current_policy

                    # Record the reward breakdown statistics
                    step_info.reward_distance = reward_distance
                    step_info.reward_roll = reward_roll
                    step_info.reward_current = reward_current
                    step_info.reward_yaw = reward_yaw
                    step_info.reward_pitch = reward_pitch
                    step_info.reward_action_smoothness = reward_action_smoothness
                    step_info.step_reward_total = float(step_performance_reward)
                    step_info.episode_reward_total = float(episode_performance_reward)

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
                reward=reward_env,
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

                # In the simulation, so the robot resets automatically
                if self.env.settings.robot.sim:
                    log.info("Skipping between-episode pause because simulation mode is active.")
                
                # In the real world, so the robot needs to be reset manually
                else:
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
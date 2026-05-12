"""
Observation preprocessing utilities for recurrent ONNX locomotion policies.

This module provides preprocessing components used before policy inference,
including support for recurrent actor variants that prepend the previous
action vector to the current observation.

This file was adapted from:

https://github.com/real-world-drl/esp-dl-quant-icra2026

Original project licensed under the MIT License.
"""

from __future__ import annotations
import logging
import numpy as np


log = logging.getLogger(__name__)

class AddActionsPreprocessor():
    """
    Prepend the previous action vector to the current observation.

    Used for recurrent RA-TD3 actor policies that expect the input format:

        [previous_action, observation]

    before ONNX policy inference.
    """

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    @property
    def output_size_extra(self) -> int:
        return self.action_dim

    def process(self, observation: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
        if prev_action is None or len(prev_action) == 0:
            prev_action = np.zeros(self.action_dim, dtype=np.float32)
        return np.concatenate(
            [np.asarray(prev_action, dtype=np.float32),
             np.asarray(observation, dtype=np.float32)],
        )
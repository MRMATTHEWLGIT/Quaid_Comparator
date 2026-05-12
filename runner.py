"""
ONNX policy runners for recurrent locomotion inference and comparator-based
policy switching in the Quaid environment.

This module provides lightweight wrappers around ONNX recurrent actor models,
including support for externally managed hidden states to enable smooth
multi-policy transitions during comparator execution.

Portions of the runner structure and ONNX inference workflow were adapted
from:

https://github.com/real-world-drl/esp-dl-quant-icra2026

Original project licensed under the MIT License.
"""

from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort


log = logging.getLogger(__name__)


class OnnxRnnPolicyRunner():
    """
    Stateless ONNX recurrent actor runner.

    The runner owns the ONNX inference session and input/output names, but the
    recurrent hidden state is supplied by the caller. This allows multiple
    policy actors to share one continuous hidden state during comparator-based
    policy switching.
    """

    OBS_INPUT = "observations"
    HT_INPUT = "h_t_in"
    ACTION_OUTPUT = "action"
    HT_OUTPUT = "h_t"

    def __init__(
        self,
        model_path: str,
        *,
        rnn_layers: int = 3,
        rnn_hidden_size: int = 64,
    ) -> None:

        # Store the RNN layers and hidden size
        self.rnn_layers = rnn_layers
        self.rnn_hidden_size = rnn_hidden_size

        # Log the loading of the ONNX model
        log.info(
            "Loading ONNX recurrent actor: %s (layers=%d, hidden=%d)",
            model_path,
            rnn_layers,
            rnn_hidden_size,
        )

        # Create the ONNX inference session
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        # Get the input and output names of the ONNX model
        input_names = [input_info.name for input_info in self._session.get_inputs()]
        output_names = [output_info.name for output_info in self._session.get_outputs()]

        # Configure the input and output names
        self._obs_input = (self.OBS_INPUT if self.OBS_INPUT in input_names else input_names[0])
        self._ht_input = (self.HT_INPUT if self.HT_INPUT in input_names else input_names[1])
        self._action_output = (self.ACTION_OUTPUT if self.ACTION_OUTPUT in output_names else output_names[0])
        self._ht_output = (self.HT_OUTPUT if self.HT_OUTPUT in output_names else output_names[1])


    def select_action(self, state: np.ndarray, h_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run one recurrent ONNX policy inference step.

        Returns the selected action and updated hidden state.
        """

        # Reshape the state to a 1D array
        x = np.asarray(state, dtype=np.float32).reshape(1, -1)

        # Run the ONNX session
        action, next_h_t = self._session.run(
            [self._action_output, self._ht_output],
            {self._obs_input: x, self._ht_input: h_t},
        )

        # Reshape the action and next hidden state to 1D arrays
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        next_h_t = np.asarray(next_h_t, dtype=np.float32)

        return action, next_h_t


    def close(self) -> None:
        """Release the ONNX inference session."""
        self._session = None
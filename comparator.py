"""
Runtime comparator for ONNX-based policy selection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from collections import deque

import numpy as np
import onnxruntime as ort
from sklearn.neighbors import NearestNeighbors

from stats import ComparatorStepInfo


log = logging.getLogger(__name__)


@dataclass
class ComparatorLookupTable:
    """
    Precomputed local statistics for each historical comparator database point.

    All arrays are aligned to database row indices.
    """

    ordered_conditions: np.ndarray
    neighbour_indices: np.ndarray
    local_condition_distributions: np.ndarray
    local_reward_mean: np.ndarray
    nearest_neighbour_index: NearestNeighbors


@dataclass
class QueryStatistics:
    """
    Comparator-visible statistics for one live query point.

    The query is embedded using the runtime GRU encoder, projected into the
    fixed UMAP space, and compared against the historical comparator database.
    """

    query_embedding: np.ndarray
    query_embedding_2d: np.ndarray
    query_policy: str
    query_step: int
    local_condition_distribution: np.ndarray
    local_reward_mean: float
    neighbour_indices: np.ndarray


@dataclass
class CandidateMaskResult:
    """
    Candidate filtering result aligned to database row indices.
    """

    candidate_mask: np.ndarray
    counts: dict[str, int]
    distribution_overlap: Optional[np.ndarray] = None


@dataclass
class CandidateVoteResult:
    """
    Candidate voting result for policy selection.

    candidate_indices are database row indices that passed the candidate mask.
    policy_vote_counts maps policy key to number of valid candidate points.
    """

    candidate_indices: np.ndarray
    policy_vote_counts: dict[str, int]
    selected_policy: str
    selected_policy_count: int
    selected_policy_fraction: float


class Comparator:
    """
    Runtime policy comparator.

    Loads the GRU embedding encoder, normalisation statistics, UMAP projection
    model, and historical comparator database from a generated comparator asset
    directory.
    """

    GRU_ONNX_NAME = "embedding_gru_encoder.onnx"
    GRU_STATS_NAME = "embedding_gru_stats.npz"
    DATABASE_NAME = "comparator_database.npz"
    METADATA_NAME = "metadata.json"

    PARAMETRIC_UMAP_ONNX_NAME = "parametric_umap_encoder.onnx"
    STANDARD_UMAP_NAME = "umap_model.pkl"

    def __init__(
        self,
        comparator_assets_dir: str,
        *,
        comparator_hyperparameters: dict,
        valid_policy_keys: Optional[set[str]] = None,
        providers: Optional[list[str]] = None,
    ) -> None:

        # Store the asset directory
        self.assets_dir = Path(comparator_assets_dir)

        if not self.assets_dir.exists():
            raise FileNotFoundError(
                f"Comparator assets directory does not exist: {self.assets_dir}"
            )

        if not self.assets_dir.is_dir():
            raise NotADirectoryError(
                f"Comparator assets path is not a directory: {self.assets_dir}"
            )

        # Use CPU by default for Raspberry Pi friendly deployment
        self.providers = providers or ["CPUExecutionProvider"]

        # Store the comparator hyperparameters
        self.k_signature = int(comparator_hyperparameters.get("k_signature", 50))
        self.k_reward = int(comparator_hyperparameters.get("k_reward", 50))
        self.sequence_length = int(comparator_hyperparameters.get("sequence_length", 99))
        self.min_reward_gain_percent = float(comparator_hyperparameters.get("min_reward_gain_percent", 0.0))
        self.min_distribution_overlap = comparator_hyperparameters.get("min_distribution_overlap", None)
        self.min_pairwise_agreement = comparator_hyperparameters.get("min_pairwise_agreement", None)
        self.include_current_policy_candidates = bool(comparator_hyperparameters.get("include_current_policy_candidates", True))
        self.min_vote_candidates = int(comparator_hyperparameters.get("min_vote_candidates", 1))
        self.min_vote_fraction = float(comparator_hyperparameters.get("min_vote_fraction", 0.0))

        # Validate the comparator hyperparameters
        if self.k_signature <= 0:
            raise ValueError("k_signature must be greater than zero.")

        if self.k_reward <= 0:
            raise ValueError("k_reward must be greater than zero.")

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero.")

        if self.min_reward_gain_percent < 0.0:
            raise ValueError("min_reward_gain_percent must be greater than or equal to zero.")

        if self.min_distribution_overlap is not None and (self.min_distribution_overlap < 0.0 or self.min_distribution_overlap > 1.0):
            raise ValueError("min_distribution_overlap must be between 0.0 and 1.0.")
        if self.min_pairwise_agreement is not None and (
            self.min_pairwise_agreement < 0.0 or self.min_pairwise_agreement > 1.0
        ):
            raise ValueError("min_pairwise_agreement must be between 0.0 and 1.0.")

        if self.min_vote_candidates < 0:
            raise ValueError("min_vote_candidates must be greater than or equal to zero.")

        if self.min_vote_fraction < 0.0 or self.min_vote_fraction > 1.0:
            raise ValueError("min_vote_fraction must be between 0.0 and 1.0.")

        # Store the set of valid policy keys
        self.valid_policy_keys = set(valid_policy_keys or [])

        # Build expected asset paths
        self.gru_onnx_path = self.assets_dir / self.GRU_ONNX_NAME
        self.gru_stats_path = self.assets_dir / self.GRU_STATS_NAME
        self.database_path = self.assets_dir / self.DATABASE_NAME
        self.metadata_path = self.assets_dir / self.METADATA_NAME

        self.parametric_umap_path = self.assets_dir / self.PARAMETRIC_UMAP_ONNX_NAME
        self.standard_umap_path = self.assets_dir / self.STANDARD_UMAP_NAME

        # Validate required files
        self._require_file(self.gru_onnx_path)
        self._require_file(self.gru_stats_path)
        self._require_file(self.database_path)
        self._require_file(self.metadata_path)

        # Load metadata
        self.metadata = self._load_metadata(self.metadata_path)

        # Load GRU normalisation statistics
        self.gru_stats = np.load(self.gru_stats_path)

        # Define the mean and standard deviation of the GRU input features
        self.X_mean = np.asarray(self.gru_stats["X_mean"], dtype=np.float32)
        self.X_std = np.asarray(self.gru_stats["X_std"], dtype=np.float32)

        # Determine expected GRU input shape from saved normalisation stats
        self.input_dim = int(np.prod(self.X_mean.shape[-1:]))

        # Live query history stores obs_t + action_t states for the embedding GRU
        self.query_history = deque(maxlen=self.sequence_length)

        # Load historical comparator database
        self.database = np.load(self.database_path, allow_pickle=True)

        # Load the embeddings and UMAP embeddings from the database
        self.embeddings = np.asarray(self.database["embeddings"], dtype=np.float32)
        self.umap_embeddings = np.asarray(self.database["umap_embeddings"], dtype=np.float32)

        # Load database metadata
        self.rewards = np.asarray(self.database["rewards"], dtype=np.float32)
        self.databases = np.asarray(self.database["databases"])
        self.episodes = np.asarray(self.database["episodes"])
        self.terrains = np.asarray(self.database["terrains"])

        n_points = len(self.embeddings)

        # Validate the size of the database arrays
        if len(self.umap_embeddings) != n_points:
            raise ValueError("UMAP embeddings size mismatch.")

        if len(self.rewards) != n_points:
            raise ValueError("Reward array size mismatch.")

        if len(self.databases) != n_points:
            raise ValueError("Database array size mismatch.")

        if len(self.episodes) != n_points:
            raise ValueError("Episode array size mismatch.")

        if len(self.terrains) != n_points:
            raise ValueError("Terrain array size mismatch.")

        # Build readable labels from terrain IDs
        self._build_database_labels()

        # Validate the policy keys in the database
        if self.valid_policy_keys:
            unknown_policy_keys = sorted(
                set(str(policy_key) for policy_key in self.policy_keys)
                - self.valid_policy_keys
            )

            if unknown_policy_keys:
                raise ValueError(
                    f"Comparator database contains unknown policy keys: {unknown_policy_keys}"
                )

        self.last_step_info = None

        # Build precomputed nearest-neighbour lookup table
        self.lookup_table = self._build_lookup_table(k_signature=self.k_signature, 
                k_reward=self.k_reward)

        # Load ONNX GRU embedding encoder
        log.info("Loading embedding GRU encoder: %s", self.gru_onnx_path)

        # Create the ONNX inference session for the GRU embedding encoder
        self.gru_session = ort.InferenceSession(
            str(self.gru_onnx_path),
            providers=self.providers,
        )

        self.gru_input_name = self.gru_session.get_inputs()[0].name
        self.gru_output_name = self.gru_session.get_outputs()[0].name

        # Load UMAP projection model
        self.umap_kind = self._detect_umap_kind()

        # Load the parametric UMAP encoder
        if self.umap_kind == "parametric":
            log.info("Loading parametric UMAP encoder: %s", self.parametric_umap_path)

            self.umap_session = ort.InferenceSession(
                str(self.parametric_umap_path),
                providers=self.providers,
            )

            self.umap_input_name = self.umap_session.get_inputs()[0].name
            self.umap_output_name = self.umap_session.get_outputs()[0].name
            self.umap_model = None

        # Load the standard UMAP encoder
        else:
            log.info("Loading standard UMAP model: %s", self.standard_umap_path)

            import pickle

            with open(self.standard_umap_path, "rb") as file:
                self.umap_model = pickle.load(file)

            self.umap_session = None
            self.umap_input_name = None
            self.umap_output_name = None

        log.info(
            "Loaded comparator assets from %s with %d database points.",
            self.assets_dir,
            len(self.umap_embeddings),
        )

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _require_file(self, path: Path) -> None:
        """Raise a clear error if a required asset file is missing."""

        if not path.exists():
            raise FileNotFoundError(f"Required comparator asset is missing: {path}")

    def _load_metadata(self, path: Path) -> dict:
        """Load comparator metadata JSON."""

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _detect_umap_kind(self) -> str:
        """Detect whether this comparator uses parametric or standard UMAP."""

        if self.parametric_umap_path.exists():
            return "parametric"

        if self.standard_umap_path.exists():
            return "standard"

        raise FileNotFoundError(
            "No UMAP projection model found. Expected either "
            f"{self.parametric_umap_path.name} or {self.standard_umap_path.name}."
        )

    def _terrain_name_from_id(
        self,
        terrain_id: int,
        idx_to_terrain: dict,
    ) -> str:
        """
        Convert a terrain ID into its saved terrain name.
        """

        terrain_key = str(int(terrain_id))

        if terrain_key not in idx_to_terrain:
            raise KeyError(f"Terrain ID {terrain_id} was not found in idx_to_terrain.")

        return str(idx_to_terrain[terrain_key])


    def _condition_name_from_terrain_name(self, terrain_name: str) -> str:
        """
        Extract the condition suffix from a terrain name.
        """

        terrain_name = str(terrain_name).lower().strip()

        if "-" not in terrain_name:
            raise ValueError(
                f"Could not determine condition name from terrain name: {terrain_name}"
            )

        return terrain_name.rsplit("-", maxsplit=1)[-1]


    def _policy_key_from_terrain_name(self, terrain_name: str) -> str:
        """
        Extract the runtime policy key prefix from a terrain name.
        """

        terrain_name = str(terrain_name).lower().strip()

        if "+" not in terrain_name:
            raise ValueError(
                f"Could not determine policy key from terrain name: {terrain_name}"
            )

        return terrain_name.split("+", maxsplit=1)[0]


    def _build_database_labels(self) -> None:
        """
        Build readable terrain, condition, and policy labels for each database point.
        """

        # Extract the terrain ID to terrain-name mapping from metadata
        idx_to_terrain = self.metadata["idx_to_terrain"]

        # Convert terrain IDs into terrain names
        self.terrain_names = np.array(
            [self._terrain_name_from_id(terrain_id, idx_to_terrain)
                for terrain_id in self.terrains
            ],
            dtype=object,
        )

        # Convert terrain names into condition names
        self.condition_names = np.array(
            [
                self._condition_name_from_terrain_name(terrain_name)
                for terrain_name in self.terrain_names
            ],
            dtype=object,
        )

        # Convert terrain names into policy keys
        self.policy_keys = np.array(
            [
                self._policy_key_from_terrain_name(terrain_name)
                for terrain_name in self.terrain_names
            ],
            dtype=object,
        )

        log.info(
            "Comparator policy keys found in database: %s",
            sorted(set(str(policy_key) for policy_key in self.policy_keys)),
        )

        log.info(
            "Comparator condition names found in database: %s",
            sorted(set(str(condition_name) for condition_name in self.condition_names)),
        )

    def _distribution_from_conditions(self,
        condition_values: np.ndarray,
        ordered_conditions: np.ndarray,
    ) -> np.ndarray:
        """
        Build a probability distribution over the ordered condition list.
        """

        distribution = np.zeros(len(ordered_conditions), dtype=np.float64)

        # Map condition names to their positions in the neighbour list
        condition_to_position = {
            str(condition_name): position
            for position, condition_name in enumerate(ordered_conditions)
        }

        # Count the number of times each condition appears in the neighbour list
        for condition_value in condition_values:
            position = condition_to_position.get(str(condition_value))
            if position is not None:
                distribution[position] += 1.0

        # Sum each value in the distribution to get a total
        total = float(distribution.sum())
        if total > 0.0:
            distribution /= total

        return distribution

    
    def _build_lookup_table(self, *, k_signature: int = 50, k_reward: int = 50) -> ComparatorLookupTable:
        """
        Precompute local neighbour statistics for every database point.

        This avoids recomputing local condition distributions and local reward
        means during runtime inference.
        """

        n_points = len(self.umap_embeddings)

        if n_points == 0:
            raise ValueError("Cannot build comparator lookup table with zero database points.")

        # Get a stable ordered list of known condition names
        ordered_conditions = np.array(
            sorted(np.unique(self.condition_names).astype(str)),
            dtype=object,
        )

        n_conditions = len(ordered_conditions)

        # Neighbour count includes the point itself, which is removed later
        neighbour_count = min(max(k_signature, k_reward, 1) + 1, n_points)

        # Fit nearest-neighbour index over the saved 2D UMAP database
        nearest_neighbour_index = NearestNeighbors(
            n_neighbors=neighbour_count,
            metric="euclidean",
        )

        # Fit the nearest-neighbour index over the saved 2D UMAP database
        nearest_neighbour_index.fit(self.umap_embeddings)

        # Query neighbours for every database point
        _, neighbour_indices_full = nearest_neighbour_index.kneighbors(
            self.umap_embeddings,
            n_neighbors=neighbour_count,
            return_distance=True,
        )

        # Store fixed-size neighbour indices excluding the point itself
        max_k = max(k_signature, k_reward, 1)
        neighbour_indices = np.full((n_points, max_k), -1, dtype=np.int64)

        # Preallocate local condition distributions
        local_condition_distributions = np.zeros(
            (n_points, n_conditions),
            dtype=np.float64,
        )

        # Preallocate local reward means
        local_reward_mean = np.zeros(n_points, dtype=np.float64)

        # Build the lookup table
        for point_idx in range(n_points):

            # Remove the current point from its own neighbour list
            neighbours = neighbour_indices_full[point_idx]
            neighbours = neighbours[neighbours != point_idx]

            # Fallback for tiny databases
            if len(neighbours) == 0:
                neighbours = np.array([point_idx], dtype=np.int64)

            # Store up to max_k neighbours in the lookup table
            stored_neighbours = neighbours[:max_k]
            neighbour_indices[point_idx, :len(stored_neighbours)] = stored_neighbours

            # Use K_SIGNATURE neighbours for local condition distribution
            signature_neighbours = neighbours[:k_signature]

            if len(signature_neighbours) == 0:
                signature_neighbours = np.array([point_idx], dtype=np.int64)

            # Build the local condition distribution for current point
            local_condition_distributions[point_idx] = self._distribution_from_conditions(
                self.condition_names[signature_neighbours],
                ordered_conditions,
            )

            # Use K_REWARD neighbours for local reward mean
            reward_neighbours = neighbours[:k_reward]

            if len(reward_neighbours) == 0:
                reward_neighbours = np.array([point_idx], dtype=np.int64)

            # Calculate the local reward mean for current point
            local_reward_mean[point_idx] = float(
                np.mean(self.rewards[reward_neighbours], dtype=np.float64)
            )

        return ComparatorLookupTable(
            ordered_conditions=ordered_conditions,
            neighbour_indices=neighbour_indices,
            local_condition_distributions=local_condition_distributions,
            local_reward_mean=local_reward_mean,
            nearest_neighbour_index=nearest_neighbour_index,
        )


    @staticmethod
    def distribution_overlap(
        query_distribution: np.ndarray,
        candidate_distributions: np.ndarray,
    ) -> np.ndarray:
        """
        Compute sum(min(p_i, q_i)) overlap between query and candidates.
        """

        # Ensure the distributions are numpy arrays
        query_distribution = np.asarray(query_distribution, dtype=np.float64)
        candidate_distributions = np.asarray(candidate_distributions, dtype=np.float64)

        if candidate_distributions.ndim == 1:
            candidate_distributions = candidate_distributions.reshape(1, -1)

        # Calculate the overlap score between the query and candidate distributions
        return np.sum(
            np.minimum(candidate_distributions, query_distribution.reshape(1, -1)),
            axis=1,
        )

    @staticmethod
    def pairwise_distribution_agreement(
        query_distribution: np.ndarray,
        candidate_distributions: np.ndarray,
        tolerance: float = 1e-12,
    ) -> np.ndarray:
        """
        Measure whether candidate condition ranking matches the query ranking.
        """

        # Ensure the distributions are numpy arrays
        query_distribution = np.asarray(query_distribution, dtype=np.float64)
        candidate_distributions = np.asarray(candidate_distributions, dtype=np.float64)

        if candidate_distributions.ndim == 1:
            candidate_distributions = candidate_distributions.reshape(1, -1)

        # Determine how many conditions in the query distribution
        n_conditions = len(query_distribution)

        # Initialise all scores to 1.0
        scores = np.ones(len(candidate_distributions), dtype=np.float64)

        # Not enough conditions to have meaningful pairwise agreement
        if n_conditions < 2:
            return scores

        # Loop over each candidate point distribution
        for candidate_position, candidate_distribution in enumerate(candidate_distributions):

            agreements = 0
            total_pairs = 0

            # Generate all unique pairs for this candidate distribution
            for first_position in range(n_conditions):
                for second_position in range(first_position + 1, n_conditions):

                    # Compare query ordering for this pair
                    query_difference = (
                        query_distribution[first_position]
                        - query_distribution[second_position]
                    )

                    # Compare candidate ordering for this pair
                    candidate_difference = (
                        candidate_distribution[first_position]
                        - candidate_distribution[second_position]
                    )

                    # Determine query order for this pair
                    query_sign = 0
                    if abs(query_difference) > tolerance:
                        query_sign = 1 if query_difference > 0 else -1

                    # Determine candidate order for this pair
                    candidate_sign = 0
                    if abs(candidate_difference) > tolerance:
                        candidate_sign = 1 if candidate_difference > 0 else -1

                    # If the query and candidate pair ordering agree, count the result
                    if query_sign == candidate_sign:
                        agreements += 1

                    total_pairs += 1

            # Calculate the pairwise agreement for this candidate distribution vs query
            scores[candidate_position] = (
                agreements / total_pairs
                if total_pairs
                else 1.0
            )

        return scores


    def _normalise_query_sequence(self) -> np.ndarray:
        """
        Build and normalise the current live query sequence.
        """

        if not self.has_enough_history():
            raise ValueError(
                f"Comparator needs {self.sequence_length} states before inference, "
                f"but only has {len(self.query_history)}."
            )

        # Stack recent obs_t + action_t states into [sequence_length, input_dim]
        sequence = np.stack(list(self.query_history), axis=0).astype(np.float32)

        # Normalise using saved GRU statistics
        sequence = (sequence - self.X_mean) / (self.X_std + 1e-8)

        # Add batch dimension for ONNX GRU encoder
        return sequence.reshape(1, self.sequence_length, self.input_dim).astype(np.float32)

    
    def _embed_query_sequence(self) -> np.ndarray:
        """
        Embed the current live query sequence using the ONNX GRU encoder.
        """

        # Build normalised GRU input
        sequence = self._normalise_query_sequence()

        # Run embedding GRU ONNX inference
        embedding = self.gru_session.run(
            [self.gru_output_name],
            {self.gru_input_name: sequence},
        )[0]

        return np.asarray(embedding, dtype=np.float32).reshape(-1)


    def _project_query_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project one GRU embedding into the fixed UMAP space.
        """

        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        # Parametric UMAP ONNX path
        if self.umap_session is not None:
            embedding_2d = self.umap_session.run(
                [self.umap_output_name],
                {self.umap_input_name: embedding},
            )[0]

        # Standard UMAP pickle path
        else:
            embedding_2d = self.umap_model.transform(embedding)

        return np.asarray(embedding_2d, dtype=np.float64).reshape(-1)


    # -----------------------------------------------------------------------
    # Runtime inference methods
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset live comparator query history at the start of an episode.
        """
        self.query_history.clear()


    def update_query_history(self, state: np.ndarray) -> None:
        """
        Add one obs_t + action_t state to the live comparator history.
        """

        # Convert to consistent runtime format
        state = np.asarray(state, dtype=np.float32).reshape(-1)

        if state.shape[0] != self.input_dim:
            raise ValueError(
                f"Comparator state has incorrect size. Expected {self.input_dim}, "
                f"got {state.shape[0]}."
            )

        self.query_history.append(state)

    
    def has_enough_history(self) -> bool:
        """
        Return whether the live query history is long enough for GRU inference.
        """
        return len(self.query_history) >= self.sequence_length


    def compute_query_statistics(
        self,
        *,
        current_policy: str,
        step: int,
    ) -> Optional[QueryStatistics]:
        """
        Compute Comparator-visible query statistics.

        The query is transformed into the fixed historical UMAP space. The query is
        not used to fit the UMAP or update the memory nearest-neighbour model.
        """

        # Not enough live history yet to build a full GRU sequence
        if not self.has_enough_history():
            return None

        # Embed the live obs_t + action_t sequence using the runtime GRU
        query_embedding = self._embed_query_sequence()

        # Transform the query point into the fixed 2D UMAP space
        query_embedding_2d = self._project_query_embedding(query_embedding)

        # Set the number of neighbours for the memory space per point
        neighbour_count = min(
            max(self.k_signature, self.k_reward, 1),
            len(self.umap_embeddings),
        )

        # Determine all neighbours within the 2D space around the transformed query point
        neighbour_indices = self.lookup_table.nearest_neighbour_index.kneighbors(
            query_embedding_2d.reshape(1, -1),
            n_neighbors=neighbour_count,
            return_distance=False,
        )[0]

        # Define all the neighbours for both distribution signature and reward calculations
        signature_neighbours = neighbour_indices[:self.k_signature]
        reward_neighbours = neighbour_indices[:self.k_reward]

        # Calculate the neighbourhood distribution around the query point
        local_condition_distribution = self._distribution_from_conditions(
            self.condition_names[signature_neighbours],
            self.lookup_table.ordered_conditions,
        )

        # Calculate the local neighbourhood reward mean around the query point
        local_reward_mean = float(np.mean(self.rewards[reward_neighbours]))

        # Create the QueryStatistics and return it
        return QueryStatistics(
            query_embedding=np.asarray(query_embedding, dtype=np.float32),
            query_embedding_2d=np.asarray(query_embedding_2d, dtype=np.float64),
            query_policy=str(current_policy),
            query_step=int(step),
            local_condition_distribution=local_condition_distribution,
            local_reward_mean=local_reward_mean,
            neighbour_indices=np.asarray(neighbour_indices, dtype=np.int64),
        )


    def build_candidate_mask(
        self,
        query_stats: QueryStatistics,
    ) -> CandidateMaskResult:
        """
        Build the candidate mask for Comparator policy selection.
        """

        # Debugger to store candidate counts after each filter stage
        counts = {}

        # The initial starting amount of memory points
        n_memory_points = len(self.umap_embeddings)

        # Set-up mask and store initial count of points
        candidate_mask = np.ones(n_memory_points, dtype=bool)
        counts["all_memory_points"] = int(candidate_mask.sum())

        # Optionally remove points from the current policy.
        if not self.include_current_policy_candidates:
            candidate_mask &= self.policy_keys != query_stats.query_policy
            counts["after_different_policy"] = int(candidate_mask.sum())

        else:
            counts["after_policy_filter"] = int(candidate_mask.sum())

        # Calculate the reward margin for the query
        reward_margin = abs(query_stats.local_reward_mean) * (
            self.min_reward_gain_percent / 100.0
        )

        # Set-up mask to allow only points that have a local reward mean greater
        # than or equal to the query
        candidate_mask &= (
            self.lookup_table.local_reward_mean
            >= query_stats.local_reward_mean + reward_margin
        )
        counts["after_reward_filter"] = int(candidate_mask.sum())

        distribution_overlap = None

        # Optional minimum distribution overlap filter
        if self.min_distribution_overlap is not None:

            # Calculate the overlap between the query and memory points
            distribution_overlap = self.distribution_overlap(
                query_stats.local_condition_distribution,
                self.lookup_table.local_condition_distributions,
            )

            # Apply the minimum distribution overlap filter
            candidate_mask &= distribution_overlap >= float(self.min_distribution_overlap)
            counts["after_distribution_overlap_filter"] = int(candidate_mask.sum())

        # Optional minimum pairwise distribution agreement filter
        if self.min_pairwise_agreement is not None:

            # Calculate the pairwise agreement between the query and memory points
            pairwise_agreement = self.pairwise_distribution_agreement(
                query_stats.local_condition_distribution,
                self.lookup_table.local_condition_distributions,
            )

            candidate_mask &= pairwise_agreement >= float(self.min_pairwise_agreement)
            counts["after_pairwise_agreement_filter"] = int(candidate_mask.sum())

        return CandidateMaskResult(
            candidate_mask=candidate_mask,
            counts=counts,
            distribution_overlap=distribution_overlap,
        )


    def vote_for_policy(
        self,
        *,
        current_policy: str,
        candidate_mask: np.ndarray,
    ) -> CandidateVoteResult:
        """
        Select a policy by counting how many valid candidate points belong to each
        policy.

        Ties are resolved by keeping the current policy if it is tied for first.
        Otherwise, the tied policy with the highest mean local reward is selected.
        """

        # Get the database indices of the candidate points
        candidate_indices = np.where(candidate_mask)[0]

        # No valid candidates exist, so return empty vote result
        if len(candidate_indices) == 0:
            return CandidateVoteResult(
                candidate_indices=np.array([], dtype=np.int64),
                policy_vote_counts={},
                selected_policy=str(current_policy),
                selected_policy_count=0,
                selected_policy_fraction=0.0,
            )

        # Get the policies of the candidate points
        candidate_policies = self.policy_keys[candidate_indices].astype(str)

        # Count the number of candidate points for each policy
        policy_vote_counts = {
            str(policy): int(np.sum(candidate_policies == str(policy)))
            for policy in sorted(np.unique(candidate_policies))
        }

        # Get the policy with the most candidate points
        max_count = max(policy_vote_counts.values())

        # Get the policies that are tied for the most candidate points
        tied_policies = [
            policy
            for policy, count in policy_vote_counts.items()
            if count == max_count
        ]

        # Prefer staying with the current policy if it is tied for first.
        if str(current_policy) in tied_policies:
            selected_policy = str(current_policy)

        else:
            # Otherwise, break ties using mean local reward among candidates
            # belonging to each tied policy.
            tied_policy_reward_means = {}

            for policy in tied_policies:

                # Get the database indices of the candidate points for the current policy
                policy_indices = candidate_indices[candidate_policies == policy]

                # Calculate the mean local reward for the current policy
                tied_policy_reward_means[policy] = float(
                    np.mean(self.lookup_table.local_reward_mean[policy_indices])
                )

            # Get the policy with the highest mean local reward
            selected_policy = max(
                tied_policy_reward_means,
                key=tied_policy_reward_means.get,
            )

        # Get the number of candidate points for the selected policy
        selected_policy_count = int(policy_vote_counts[selected_policy])

        # Calculate the fraction of candidate points for the selected policy
        selected_policy_fraction = float(selected_policy_count / len(candidate_indices))

        # Create and return the candidate vote result
        return CandidateVoteResult(
            candidate_indices=np.asarray(candidate_indices, dtype=np.int64),
            policy_vote_counts=policy_vote_counts,
            selected_policy=selected_policy,
            selected_policy_count=selected_policy_count,
            selected_policy_fraction=selected_policy_fraction,
        )

    def select_policy(
        self,
        *,
        current_policy: str,
        step: int,
    ) -> str:
        f"""
        Select the policy to use at the next timestep.

        The comparator:
        1. embeds the live obs_(t + 1) + action_t history,
        2. projects the query into UMAP space,
        3. computes query-local statistics,
        4. builds a candidate mask,
        5. counts valid candidates per policy,
        6. returns the policy key with the strongest candidate support.

        If the comparator cannot make a valid decision, the current policy is kept.
        """

        # Default debug info for this step
        self.last_step_info = ComparatorStepInfo(
            step=int(step),
            current_policy=str(current_policy),
            next_policy=str(current_policy),
            switch_committed=False,
            can_switch=False,
            candidate_count=0,
            query_local_reward_mean=None,
            query_umap_x=None,
            query_umap_y=None,
            candidate_indices_json=None,
            policy_vote_counts_json=None,
            selected_policy_count=None,
            selected_policy_fraction=None,
            candidate_filter_counts_json=None,
        )

        # Compute the live query statistics from the current history buffer
        query_stats = self.compute_query_statistics(
            current_policy=current_policy,
            step=step,
        )

        # Not enough history yet, so keep using the current policy
        if query_stats is None:
            return current_policy

        self.last_step_info.query_local_reward_mean = float(
            query_stats.local_reward_mean
        )

        # Store the query's 2D UMAP coordinates for logging and analysis
        self.last_step_info.query_umap_x = float(query_stats.query_embedding_2d[0])
        self.last_step_info.query_umap_y = float(query_stats.query_embedding_2d[1])

        # Build the candidate mask based on query-vs-memory criteria
        candidate_mask_result = self.build_candidate_mask(query_stats)
        self.last_step_info.candidate_filter_counts_json = json.dumps(candidate_mask_result.counts)

        # Vote over all valid candidate points
        vote_result = self.vote_for_policy(
            current_policy=current_policy,
            candidate_mask=candidate_mask_result.candidate_mask,
        )

        self.last_step_info.candidate_count = int(len(vote_result.candidate_indices))

        # Not enough candidates to vote, so keep using the current policy
        if len(vote_result.candidate_indices) < self.min_vote_candidates:
            return current_policy

        # Not enough candidate support for the selected policy, so keep using the current policy
        if vote_result.selected_policy_fraction < self.min_vote_fraction:
            return current_policy

        # No valid candidates exist, so keep using the current policy
        if len(vote_result.candidate_indices) == 0:
            log.debug(
                "Comparator step=%d current=%s no valid candidates counts=%s",
                step,
                current_policy,
                candidate_mask_result.counts,
            )

            return current_policy

        # Get the selected policy from the vote result
        next_policy = str(vote_result.selected_policy)

        # Store statistics to keep track of the comparator's decision
        self.last_step_info.next_policy = next_policy
        self.last_step_info.candidate_count = int(len(vote_result.candidate_indices))
        self.last_step_info.candidate_indices_json = json.dumps(vote_result.candidate_indices.astype(int).tolist())
        self.last_step_info.policy_vote_counts_json = json.dumps(vote_result.policy_vote_counts)
        self.last_step_info.selected_policy_count = int(vote_result.selected_policy_count)
        self.last_step_info.selected_policy_fraction = float(vote_result.selected_policy_fraction)

        # Log the comparator's decision
        log.debug(
            "Comparator vote step=%d current=%s next=%s candidates=%d votes=%s "
            "selected_count=%d selected_fraction=%.3f counts=%s",
            step,
            current_policy,
            next_policy,
            len(vote_result.candidate_indices),
            vote_result.policy_vote_counts,
            vote_result.selected_policy_count,
            vote_result.selected_policy_fraction,
            candidate_mask_result.counts,
        )

        return next_policy


    def close(self) -> None:
        """Release comparator ONNX sessions."""
        self.gru_session = None
        self.umap_session = None
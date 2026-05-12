"""
Runtime comparator for ONNX-based policy selection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort


log = logging.getLogger(__name__)


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
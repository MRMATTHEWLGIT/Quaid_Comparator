"""
Build offline comparator assets from historical rollout data.

This script:
- extracts GRU embeddings from a rollout testset,
- exports the embedding GRU to ONNX,
- fits a standard or parametric UMAP model,
- projects all embeddings into UMAP space,
- saves the comparator database and runtime assets.

Generated assets are saved into:
    models/comparator/
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the offline comparator database and UMAP projection assets."
    )

    parser.add_argument("--gru-path", required=True, type=Path,
                        help="Path to the trained embedding GRU checkpoint (.pth).")
    parser.add_argument("--testset-path", required=True, type=Path,
                        help="Historical rollout/test dataset used for the comparator database.")
    parser.add_argument("--output-root", default=Path("models"), type=Path,
                        help="Root directory where comparator assets will be saved.")
    parser.add_argument("--umap-kind", choices=["standard", "parametric"], required=True,
                        help="Whether to fit standard UMAP or parametric UMAP.")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors parameter.")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist parameter.")
    parser.add_argument("--metric", default="euclidean", help="UMAP distance metric.")
    parser.add_argument("--parametric-epochs", type=int, default=10,
                        help="Training epochs for parametric UMAP.")
    parser.add_argument("--parametric-batch-size", type=int, default=4096,
                    help="Batch size used when training parametric UMAP.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for UMAP.")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# GRU model
# ---------------------------------------------------------------------------

class RNN_GRU(nn.Module):
    """Simple GRU model used for comparator embedding extraction."""

    def __init__(self, input_size, output_size, hidden_size, num_layers, bidirectional=False):
        super().__init__()

        # Create a GRU module
        self.gru = nn.GRU(input_size=input_size,  # F
                          hidden_size=hidden_size,  # H
                          num_layers=num_layers,
                          batch_first=True,  # Input shape: (Batch Size, seq_len, F)
                          bidirectional=bidirectional)

        # Account for bidirectional GRU options in-terms of output dimensions
        out_dim = hidden_size * (2 if bidirectional else 1)

        # Create a linear layer for the head of the GRU module
        self.head = nn.Linear(out_dim, output_size)

    def forward(self, x):

        # Output of the GRU model
        out, _ = self.gru(x)  # out - (B, T, H)

        # Compute the mean across all hidden states of the final GRU layer
        mean_state = out.mean(dim=1)

        # Last time step hidden state which summarises the entire sequence
        last = out[:, -1, :]  # last - (B, H)

        return self.head(last), mean_state


class GruEmbeddingEncoder(nn.Module):
    """Export wrapper that returns only the comparator embedding."""

    def __init__(self, model: RNN_GRU) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Forward pass through the GRU model that only outputs the mean state, 
        which is used as the comparator embedding.
        """
        _, mean_state = self.model(x)
        return mean_state


# ---------------------------------------------------------------------------
# GRU loading and export
# ---------------------------------------------------------------------------

def load_gru(gru_path: Path, device: torch.device):
    """Load the embedding GRU, normalisation statistics, and hyperparameters."""

    # Load the trained GRU model
    saved_model = torch.load(gru_path, map_location="cpu", weights_only=True)

    # Load the normalisation statistics and hyperparameters
    stats = saved_model["model_stats"]
    hp = saved_model["hyperparams"]

    # Build the GRU model
    model = RNN_GRU(input_size=hp["input_size"],
                    output_size=hp["output_size"],
                    hidden_size=hp["hidden_size"],
                    num_layers=hp["num_layers"],
                    bidirectional=hp["bidirectional"])

    # Load the model state
    model.load_state_dict(saved_model["model_state"])
    model.to(device)
    model.eval()

    return model, stats, hp


def export_embedding_gru_to_onnx(model: RNN_GRU, onnx_path: Path, input_shape, device) -> Path:
    """Export the embedding-only GRU encoder to ONNX."""

    # Create a GRU wrapper to help with the ONNX export
    encoder = GruEmbeddingEncoder(model).to(device)
    encoder.eval()

    # Use the historical dataset shape for the dummy export input
    _, sequence_length, input_size = input_shape
    dummy_x = torch.zeros(1, sequence_length, input_size, dtype=torch.float32, device=device)

    # Create the parent directory if it doesn't exist
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # Export the GRU model to ONNX
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            dummy_x,
            onnx_path,
            verbose=False,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["sequence"],
            output_names=["embedding"],
            dynamic_axes={"sequence": {0: "batch"}, "embedding": {0: "batch"}},
        )

    return onnx_path


def _to_numpy(value):
    """Convert saved tensors/statistics to NumPy arrays."""

    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def save_embedding_gru_stats(save_dir: Path, stats: tuple) -> Path:
    """Save GRU normalisation statistics for runtime inference."""

    # Extract the normalisation statistics
    X_mean, X_std, y_mean, y_std = stats

    # Save the normalisation statistics to a .npz file
    stats_path = save_dir / "embedding_gru_stats.npz"
    np.savez_compressed(stats_path,
                        X_mean=_to_numpy(X_mean),
                        X_std=_to_numpy(X_std),
                        y_mean=_to_numpy(y_mean),
                        y_std=_to_numpy(y_std))

    return stats_path


# ---------------------------------------------------------------------------
# GRU embedding extraction
# ---------------------------------------------------------------------------

def load_testset(testset_path: Path, stats: tuple):
    """Load and normalise the historical comparator dataset."""

    # Load the testset from a .npz file and extract required information
    testset = np.load(testset_path, allow_pickle=True)
    X = torch.from_numpy(testset["X"]).float()
    y = torch.from_numpy(testset["y"]).float()
    rewards = torch.from_numpy(testset["rewards"]).float()
    episode_ids = torch.from_numpy(testset["episode_ids"]).long()
    database_ids = torch.from_numpy(testset["database_ids"]).long()

    # Load terrain types and normalise all to lower case
    raw_terrains = testset["terrain_ids"]
    norm_terrains = np.array([str(t).strip().lower() for t in raw_terrains])

    # Get the unique terrain values and create a map between type and int
    unique_terrains = np.unique(norm_terrains)
    terrain_map = {terrain: idx for idx, terrain in enumerate(unique_terrains)}

    # Convert all terrain types to an int ID
    terrain_ints = np.array([terrain_map[t] for t in norm_terrains], dtype=np.int64)
    terrain_ids = torch.from_numpy(terrain_ints)

    # Extract training normalisation statistics based on trained GRU model
    X_mean, X_std, y_mean, y_std = [torch.as_tensor(_to_numpy(s)).float() for s in stats]

    # Standardise the features of X and y based on the training set statistics
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std

    # Create a dataloader for the testset
    dataset = TensorDataset(X_norm, y_norm, rewards, database_ids, episode_ids, terrain_ids)
    test_dl = DataLoader(dataset, batch_size=256, shuffle=False)

    return test_dl, terrain_map, tuple(X.shape)


def extract_embeddings(model: RNN_GRU, stats: tuple, testset_path: Path, device: torch.device):
    """Run the trained embedding GRU over the historical dataset."""

    # Load the testset and extract the terrain map and input shape
    test_dl, terrain_map, input_shape = load_testset(testset_path=testset_path, stats=stats)

    # Create a map between terrain ID and terrain type
    idx_to_terrain = {int(idx): terrain for terrain, idx in terrain_map.items()}

    # Create a mean squared error loss function
    mse_loss = nn.MSELoss()

    embeds = []
    embeds_rew = []
    db_ids_all = []
    ep_ids_all = []
    terrain_ids_all = []

    total_loss = 0.0
    total_samples = 0

    # Perform inference on the testset
    with torch.no_grad():
        for xb, yb, rew, db_ids, ep_ids, terrain_ids in test_dl:

            # Move the data to the device
            xb = xb.to(device)
            yb = yb.to(device)

            # Forward pass through the GRU model
            preds, mean_state = model(xb)

            # Compute the mean squared error loss
            loss = mse_loss(preds, yb)

            # Update the total loss and number of samples
            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)

            # Append the embeddings, rewards, database IDs, episode IDs, and terrain IDs
            embeds.append(mean_state.cpu())
            embeds_rew.append(rew)
            db_ids_all.append(db_ids)
            ep_ids_all.append(ep_ids)
            terrain_ids_all.append(terrain_ids)

    # Compute the average mean squared error loss over the testset
    avg_mse = total_loss / total_samples
    print(f"Average embedding GRU test MSE: {avg_mse:.6f}")

    # Convert into useable numpy arrays
    embeddings = torch.cat(embeds, dim=0).numpy()
    rewards = torch.cat(embeds_rew, dim=0).numpy()
    databases = torch.cat(db_ids_all, dim=0).numpy()
    episodes = torch.cat(ep_ids_all, dim=0).numpy()
    terrains = torch.cat(terrain_ids_all, dim=0).numpy()

    return {
        "embeddings": embeddings,
        "rewards": rewards,
        "databases": databases,
        "episodes": episodes,
        "terrains": terrains,
        "idx_to_terrain": idx_to_terrain,
        "avg_mse": avg_mse,
        "input_shape": input_shape,
    }


# ---------------------------------------------------------------------------
# Save directory
# ---------------------------------------------------------------------------

def create_comparator_save_dir(output_root: Path, umap_kind: str) -> Path:
    """Create a timestamped comparator asset directory."""

    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = output_root / "comparator" / f"comparator_{umap_kind}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=False)

    return save_dir


# ---------------------------------------------------------------------------
# UMAP fitting
# ---------------------------------------------------------------------------

def fit_standard_umap(embeddings: np.ndarray, args: argparse.Namespace):
    """Fit standard UMAP and return the model plus projected embeddings."""

    import umap

    # Fit a standard UMAP model
    umap_model = umap.UMAP(n_neighbors=args.n_neighbors,
                           min_dist=args.min_dist,
                           metric=args.metric,
                           random_state=args.random_state)

    # Project the embeddings into the standard UMAP space
    umap_embeddings = umap_model.fit_transform(embeddings)

    return umap_model, umap_embeddings


def fit_parametric_umap(embeddings: np.ndarray, args: argparse.Namespace):
    """Fit parametric UMAP and return the model plus projected embeddings."""

    from umap.parametric_umap import ParametricUMAP

    # Train a parametric UMAP model
    parametric_umap = ParametricUMAP(batch_size=args.parametric_batch_size,
                                     n_neighbors=args.n_neighbors,
                                     min_dist=args.min_dist,
                                     metric=args.metric,
                                     random_state=args.random_state,
                                     verbose=True)

    # This version does not accept n_training_epochs in __init__.
    # Set it directly after construction.
    # This implementation displays:
    # epochs = loss_report_frequency (10 by default) * n_training_epochs
    parametric_umap.n_training_epochs = max(1, args.parametric_epochs // 10)

    # Project the embeddings into the parametric UMAP space
    umap_embeddings = parametric_umap.fit_transform(embeddings)

    return parametric_umap, umap_embeddings


def export_parametric_umap_encoder_to_onnx(parametric_umap, onnx_path: Path,
                                           input_dim: int, opset: int = 13) -> None:
    """Export the trained parametric UMAP encoder to ONNX."""

    import tensorflow as tf
    import tf2onnx

    encoder = parametric_umap.encoder

    # Wrap the encoder in a clean functional Keras model for tf2onnx
    inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name="embeddings")
    outputs = encoder(inputs)
    export_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="umap_encoder")

    # Force model build before conversion
    _ = export_model(tf.zeros((1, input_dim), dtype=tf.float32))

    # Create the input signature for the parametric UMAP encoder
    input_signature = [tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32, name="embeddings")]

    # Export the parametric UMAP encoder to ONNX
    model_proto, _ = tf2onnx.convert.from_keras(
        export_model,
        input_signature=input_signature,
        opset=opset,
        output_path=str(onnx_path),
    )

# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_comparator_database(save_dir: Path, extracted: dict, umap_embeddings: np.ndarray) -> Path:
    """Save the comparator database arrays into one compressed NPZ file."""

    database_path = save_dir / "comparator_database.npz"

    np.savez_compressed(database_path,
                        embeddings=extracted["embeddings"],
                        umap_embeddings=umap_embeddings,
                        rewards=extracted["rewards"],
                        databases=extracted["databases"],
                        episodes=extracted["episodes"],
                        terrains=extracted["terrains"])

    return database_path


def save_json(path: Path, data: dict) -> None:
    """Save a dictionary as formatted JSON."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_standard_umap_model(save_dir: Path, umap_model) -> Path:
    """Save a standard UMAP model using pickle."""

    umap_path = save_dir / "umap_model.pkl"

    with umap_path.open("wb") as file:
        pickle.dump(umap_model, file)

    return umap_path


def write_comparator_config_template(save_dir: Path, database_path: Path, umap_kind: str,
                                     umap_path: Path, gru_onnx_path: Path,
                                     stats_path: Path) -> Path:
    """Write a small YAML snippet that can be copied into comparator.yaml."""

    config_path = save_dir / "comparator_assets.yaml"

    text = f"""# Comparator assets generated offline

comparator_database_path: "{database_path.as_posix()}"

embedding_gru:
  kind: "onnx"
  path: "{gru_onnx_path.as_posix()}"
  stats_path: "{stats_path.as_posix()}"

umap:
  kind: "{umap_kind}"
  path: "{umap_path.as_posix()}"
"""

    config_path.write_text(text, encoding="utf-8")

    return config_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:

    # Get the arguments
    args = get_args(argv)

    # Create the save directory
    save_dir = create_comparator_save_dir(output_root=args.output_root, 
            umap_kind=args.umap_kind)
    print(f"Saving comparator assets to: {save_dir}")

    # Get CUDA device if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the GRU model, normalisation statistics, and hyperparameters
    model, stats, hp = load_gru(gru_path=args.gru_path, device=device)


    # Use the GRU model to extract embeddings from the testset
    extracted = extract_embeddings(model=model, stats=stats, testset_path=args.testset_path,
                                   device=device)

    # Save the normalisation statistics
    stats_path = save_embedding_gru_stats(save_dir=save_dir, stats=stats)

    # Export the GRU model to ONNX
    gru_onnx_path = export_embedding_gru_to_onnx(model=model,
                                                 onnx_path=save_dir / "embedding_gru_encoder.onnx",
                                                 input_shape=extracted["input_shape"],
                                                 device=device)

    # Convert the embeddings to a numpy array
    embeddings = extracted["embeddings"].astype(np.float32)

    # Fit and save a standard UMAP model
    if args.umap_kind == "standard":

        umap_model, umap_embeddings = fit_standard_umap(embeddings=embeddings, args=args)
        umap_path = save_standard_umap_model(save_dir=save_dir, umap_model=umap_model)
        runtime_umap_kind = "standard"

    # Train and save a parametric UMAP model
    elif args.umap_kind == "parametric":

        # Train a parametric UMAP model on the embeddings
        parametric_umap, umap_embeddings = fit_parametric_umap(embeddings=embeddings, args=args)

        # Save the parametric UMAP model to a Keras file
        keras_path = save_dir / "parametric_umap_encoder.keras"
        parametric_umap.encoder.save(keras_path)

        # Export the parametric UMAP model to ONNX
        onnx_path = save_dir / "parametric_umap_encoder.onnx"
        export_parametric_umap_encoder_to_onnx(parametric_umap=parametric_umap,
                                               onnx_path=onnx_path,
                                               input_dim=embeddings.shape[1])

        # Set the UMAP path to the ONNX model
        umap_path = onnx_path
        runtime_umap_kind = "parametric_onnx"

    else:
        raise ValueError(f"Unsupported UMAP kind: {args.umap_kind}")

    # Save the comparator database
    database_path = save_comparator_database(save_dir=save_dir,
                                             extracted=extracted,
                                             umap_embeddings=umap_embeddings.astype(np.float32))

    # Create and log all metadata
    metadata = {
        "umap_kind": args.umap_kind,
        "runtime_umap_kind": runtime_umap_kind,
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "metric": args.metric,
        "parametric_epochs": args.parametric_epochs,
        "random_state": args.random_state,
        "embedding_dim": int(embeddings.shape[1]),
        "num_points": int(embeddings.shape[0]),
        "input_shape": list(extracted["input_shape"]),
        "gru_hyperparams": hp,
        "avg_embedding_gru_mse": float(extracted["avg_mse"]),
        "idx_to_terrain": extracted["idx_to_terrain"],
        "database_path": database_path.as_posix(),
        "embedding_gru_onnx_path": gru_onnx_path.as_posix(),
        "embedding_gru_stats_path": stats_path.as_posix(),
        "umap_path": umap_path.as_posix(),
    }

    # Save the metadata to a JSON file
    save_json(path=save_dir / "metadata.json", data=metadata)

    # Write a small YAML snippet that can be copied into comparator.yaml
    config_path = write_comparator_config_template(save_dir=save_dir,
                                                   database_path=database_path,
                                                   umap_kind=runtime_umap_kind,
                                                   umap_path=umap_path,
                                                   gru_onnx_path=gru_onnx_path,
                                                   stats_path=stats_path)

    print("\nComparator database build complete.")
    print(f"Database: {database_path}")
    print(f"Embedding GRU ONNX: {gru_onnx_path}")
    print(f"Embedding GRU stats: {stats_path}")
    print(f"UMAP asset: {umap_path}")
    print(f"Metadata: {save_dir / 'metadata.json'}")
    print(f"YAML snippet: {config_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Maximum number of samples to use for the comparator database
MAX_COMPARATOR_SAMPLES = 30_000


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
# Helper functions
# ---------------------------------------------------------------------------


def policy_key_from_terrain_name(terrain_name: str) -> str:
    """
    Extract the policy key prefix from a terrain name.

    Examples:
        flat+305.334-mat    -> flat
        ramp+301.146-ramp4 -> ramp
        uneven+348.233-mat -> uneven
    """

    terrain_name = str(terrain_name).lower().strip()

    if "+" not in terrain_name:
        raise ValueError(
            f"Could not determine policy key from terrain name: {terrain_name}"
        )

    return terrain_name.split("+", maxsplit=1)[0]


def condition_name_from_terrain_name(terrain_name: str) -> str:
    """
    Extract the condition suffix from a terrain name.

    Examples:
        flat+305.334-mat    -> mat
        flat+305.334-flat   -> flat
        ramp+301.146-ramp4 -> ramp4
    """

    terrain_name = str(terrain_name).lower().strip()

    if "-" not in terrain_name:
        raise ValueError(
            f"Could not determine condition name from terrain name: {terrain_name}"
        )

    return terrain_name.rsplit("-", maxsplit=1)[-1]


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


def project_embeddings_with_umap_onnx(
    onnx_path: Path,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Project embeddings through the exported Parametric UMAP ONNX encoder."""

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    projected = session.run(
        [output_name],
        {input_name: embeddings.astype(np.float32)},
    )[0]

    return np.asarray(projected, dtype=np.float32)


def check_parametric_umap_alignment(
    saved_umap_embeddings: np.ndarray,
    onnx_umap_embeddings: np.ndarray,
) -> None:
    """Check whether saved UMAP coordinates match ONNX-projected coordinates."""

    errors = np.linalg.norm(
        saved_umap_embeddings.astype(np.float32) - onnx_umap_embeddings.astype(np.float32),
        axis=1,
    )

    print("\nParametric UMAP ONNX alignment check:")
    print(f"  mean difference:   {errors.mean():.6f}")
    print(f"  median difference: {np.median(errors):.6f}")
    print(f"  max difference:    {errors.max():.6f}")


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

def stratified_subsample_indices(
    terrain_names: np.ndarray,
    *,
    max_samples: int,
    random_state: int,
) -> tuple[np.ndarray, dict]:
    """
    Randomly subsample indices while balancing across policy-condition groups.
    """

    total_samples = int(len(terrain_names))

    # Do not need to subsample if the total number of samples is less than the maximum
    if total_samples <= max_samples:
        return np.arange(total_samples), {
            "was_subsampled": False,
            "original_num_samples": total_samples,
            "used_num_samples": total_samples,
            "max_comparator_samples": max_samples,
            "group_counts_before": {},
            "group_counts_after": {},
        }

    rng = np.random.default_rng(random_state)

    group_to_indices: dict[str, list[int]] = {}

    for index, terrain_name in enumerate(terrain_names):
        policy_key = policy_key_from_terrain_name(terrain_name)
        condition_name = condition_name_from_terrain_name(terrain_name)
        group_key = f"{policy_key}/{condition_name}"

        group_to_indices.setdefault(group_key, []).append(index)

    group_keys = sorted(group_to_indices.keys())
    num_groups = len(group_keys)

    base_quota = max_samples // num_groups
    remainder = max_samples % num_groups

    selected_indices = []
    leftover_capacity = 0

    group_counts_before = {
        group_key: len(indices)
        for group_key, indices in group_to_indices.items()
    }

    group_counts_after = {}

    # First pass: give each group an approximately equal quota.
    for group_position, group_key in enumerate(group_keys):
        group_indices = np.asarray(group_to_indices[group_key], dtype=np.int64)

        quota = base_quota + (1 if group_position < remainder else 0)
        take_count = min(quota, len(group_indices))

        chosen = rng.choice(group_indices, size=take_count, replace=False)

        selected_indices.append(chosen)
        group_counts_after[group_key] = int(take_count)

        leftover_capacity += quota - take_count

    selected_indices = np.concatenate(selected_indices)

    # Second pass: if some small groups could not fill their quota, fill the
    # remaining capacity from samples not already selected.
    if leftover_capacity > 0:
        selected_mask = np.zeros(total_samples, dtype=bool)
        selected_mask[selected_indices] = True

        remaining_indices = np.flatnonzero(~selected_mask)
        extra_count = min(leftover_capacity, len(remaining_indices))

        if extra_count > 0:
            extra_indices = rng.choice(
                remaining_indices,
                size=extra_count,
                replace=False,
            )

            selected_indices = np.concatenate([selected_indices, extra_indices])

    selected_indices = np.sort(selected_indices.astype(np.int64))

    sample_info = {
        "was_subsampled": True,
        "original_num_samples": total_samples,
        "used_num_samples": int(len(selected_indices)),
        "max_comparator_samples": max_samples,
        "group_counts_before": group_counts_before,
        "group_counts_after": {
            group_key: int(np.sum(np.isin(selected_indices, group_to_indices[group_key])))
            for group_key in group_keys
        },
    }

    return selected_indices, sample_info


def load_testset(testset_path: Path, stats: tuple, random_state: int = 42):
    """Load, stratified-subsample, and normalise the historical comparator dataset."""

    testset = np.load(testset_path, allow_pickle=True)

    X_np = testset["X"]
    y_np = testset["y"]
    rewards_np = testset["rewards"]
    episode_ids_np = testset["episode_ids"]
    database_ids_np = testset["database_ids"]
    raw_terrains = testset["terrain_ids"]

    original_num_samples = int(X_np.shape[0])

    # Normalised terrain strings are used both for stratified sampling and for
    # the final terrain ID mapping saved into metadata.
    norm_terrains = np.array([str(t).strip().lower() for t in raw_terrains])

    sample_indices, sample_info = stratified_subsample_indices(
        terrain_names=norm_terrains,
        max_samples=MAX_COMPARATOR_SAMPLES,
        random_state=random_state,
    )

    if sample_info["was_subsampled"]:
        X_np = X_np[sample_indices]
        y_np = y_np[sample_indices]
        rewards_np = rewards_np[sample_indices]
        episode_ids_np = episode_ids_np[sample_indices]
        database_ids_np = database_ids_np[sample_indices]
        raw_terrains = raw_terrains[sample_indices]
        norm_terrains = norm_terrains[sample_indices]

        print(
            f"Stratified subsampled comparator testset from "
            f"{original_num_samples:,} to {len(sample_indices):,} samples."
        )

        print("Samples per policy/condition after subsampling:")
        for group_key, count in sorted(sample_info["group_counts_after"].items()):
            print(f"  {group_key}: {count:,}")

    else:
        print(f"Using all {original_num_samples:,} comparator samples.")

    X = torch.from_numpy(X_np).float()
    y = torch.from_numpy(y_np).float()
    rewards = torch.from_numpy(rewards_np).float()
    episode_ids = torch.from_numpy(episode_ids_np).long()
    database_ids = torch.from_numpy(database_ids_np).long()

    unique_terrains = np.unique(norm_terrains)
    terrain_map = {terrain: idx for idx, terrain in enumerate(unique_terrains)}

    terrain_ints = np.array([terrain_map[t] for t in norm_terrains], dtype=np.int64)
    terrain_ids = torch.from_numpy(terrain_ints)

    X_mean, X_std, y_mean, y_std = [torch.as_tensor(_to_numpy(s)).float()for s in stats]

    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std

    dataset = TensorDataset(X_norm, y_norm, rewards, database_ids, episode_ids, terrain_ids)

    test_dl = DataLoader(dataset, batch_size=256, shuffle=False)

    return test_dl, terrain_map, tuple(X.shape), sample_info


def extract_embeddings(model: RNN_GRU, stats: tuple, testset_path: Path, device: torch.device):
    """Run the trained embedding GRU over the historical dataset."""

    # Load the testset and extract the terrain map and input shape
    test_dl, terrain_map, input_shape, sample_info = load_testset(testset_path=testset_path, stats=stats)

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
        "sample_info": sample_info,
    }



# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def safe_filename(text: str) -> str:
    """Convert arbitrary text into a safe filename component."""

    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def get_plot_labels(extracted: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build terrain, policy, and condition labels for every extracted embedding.
    """

    idx_to_terrain = {
        int(index): str(terrain)
        for index, terrain in extracted["idx_to_terrain"].items()
    }

    terrain_names = np.array(
        [idx_to_terrain[int(terrain_id)] for terrain_id in extracted["terrains"]],
        dtype=object,
    )

    policy_keys = np.array(
        [policy_key_from_terrain_name(terrain_name) for terrain_name in terrain_names],
        dtype=object,
    )

    condition_names = np.array(
        [condition_name_from_terrain_name(terrain_name) for terrain_name in terrain_names],
        dtype=object,
    )

    return terrain_names, policy_keys, condition_names


def scatter_umap_by_label(
    ax,
    umap_embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    point_size: float = 8.0,
    alpha: float = 0.55,
) -> None:
    """Draw a labelled UMAP scatter plot on an existing axis."""

    unique_labels = sorted(np.unique(labels.astype(str)))
    colour_map = plt.get_cmap("tab10")

    for label_index, label in enumerate(unique_labels):
        label_mask = labels.astype(str) == label

        ax.scatter(
            umap_embeddings[label_mask, 0],
            umap_embeddings[label_mask, 1],
            s=point_size,
            alpha=alpha,
            color=colour_map(label_index % 10),
            label=str(label),
            linewidths=0,
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, markerscale=2.0)


def plot_overall_umap_by_policy_and_condition(
    plot_dir: Path,
    umap_embeddings: np.ndarray,
    policy_keys: np.ndarray,
    condition_names: np.ndarray,
) -> Path:
    """
    Plot the full comparator UMAP space labelled by policy and condition.

    The left subplot colours points by policy. The right subplot colours the
    same points by terrain condition so the two views are easy to compare.
    """

    plot_path = plot_dir / "umap_overall_policy_vs_condition.png"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    scatter_umap_by_label(
        axes[0],
        umap_embeddings,
        policy_keys,
        title="All embeddings labelled by policy",
    )

    scatter_umap_by_label(
        axes[1],
        umap_embeddings,
        condition_names,
        title="All embeddings labelled by condition",
    )

    fig.suptitle("Comparator UMAP Space", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return plot_path


def plot_policy_umap_by_condition(
    plot_dir: Path,
    umap_embeddings: np.ndarray,
    policy_keys: np.ndarray,
    condition_names: np.ndarray,
) -> list[Path]:
    """
    Create one UMAP plot per policy, with points labelled by condition.
    """

    save_paths = []

    for policy in sorted(np.unique(policy_keys.astype(str))):
        policy_mask = policy_keys.astype(str) == policy

        if not np.any(policy_mask):
            continue

        fig, ax = plt.subplots(figsize=(8, 7))

        scatter_umap_by_label(
            ax,
            umap_embeddings[policy_mask],
            condition_names[policy_mask],
            title=f"Policy island: {policy} labelled by condition",
            point_size=10.0,
            alpha=0.65,
        )

        fig.tight_layout()

        plot_path = plot_dir / f"umap_policy_{safe_filename(policy)}_by_condition.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        save_paths.append(plot_path)

    return save_paths


def clip_reward_values(
    rewards: np.ndarray,
    clip_percentile_range: tuple[float, float] | None,
) -> np.ndarray:
    """Optionally percentile-clip reward values for clearer violin plots."""

    rewards = rewards[np.isfinite(rewards)]

    if clip_percentile_range is None or len(rewards) == 0:
        return rewards

    low_percentile, high_percentile = clip_percentile_range
    low_value, high_value = np.percentile(rewards, [low_percentile, high_percentile])

    return rewards[(rewards >= low_value) & (rewards <= high_value)]


def plot_reward_violin_by_policy(
    plot_dir: Path,
    rewards: np.ndarray,
    policy_keys: np.ndarray,
    *,
    clip_percentile_range: tuple[float, float] | None = (1.0, 99.0),
) -> Path:
    """
    Create a simple reward violin plot grouped by policy.
    """

    plot_path = plot_dir / "reward_violin_by_policy.png"

    rewards = np.asarray(rewards, dtype=np.float64)
    policy_keys = policy_keys.astype(str)

    policy_labels = []
    policy_reward_data = []

    for policy in sorted(np.unique(policy_keys)):
        policy_mask = policy_keys == policy
        policy_rewards = clip_reward_values(
            rewards[policy_mask],
            clip_percentile_range=clip_percentile_range,
        )

        if len(policy_rewards) == 0:
            continue

        policy_labels.append(policy)
        policy_reward_data.append(policy_rewards)

    if len(policy_reward_data) == 0:
        raise ValueError("No valid reward values were available for the violin plot.")

    fig, ax = plt.subplots(figsize=(10, 6))

    violin = ax.violinplot(
        policy_reward_data,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    colour_map = plt.get_cmap("tab10")

    for index, body in enumerate(violin["bodies"]):
        body.set_facecolor(colour_map(index % 10))
        body.set_edgecolor("black")
        body.set_alpha(0.8)

    positions = np.arange(1, len(policy_reward_data) + 1)

    q1_values = [np.percentile(values, 25) for values in policy_reward_data]
    median_values = [np.percentile(values, 50) for values in policy_reward_data]
    q3_values = [np.percentile(values, 75) for values in policy_reward_data]
    p5_values = [np.percentile(values, 5) for values in policy_reward_data]
    p95_values = [np.percentile(values, 95) for values in policy_reward_data]

    ax.vlines(positions, p5_values, p95_values, color="black", linewidth=1.2, alpha=0.7)
    ax.vlines(positions, q1_values, q3_values, color="black", linewidth=5.0, label="Q1 to Q3")

    ax.scatter(
        positions,
        median_values,
        color="white",
        edgecolor="black",
        zorder=3,
        label="Median",
    )

    y_min, y_max = ax.get_ylim()
    y_offset = (y_max - y_min) * 0.03

    for index, values in enumerate(policy_reward_data):
        ax.text(
            index + 1,
            min(np.max(values) + y_offset, y_max - y_offset),
            f"n={len(values)}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    title = "Reward distribution by policy"

    if clip_percentile_range is not None:
        title += f" ({clip_percentile_range[0]:.0f}-{clip_percentile_range[1]:.0f} percentile clipped)"

    ax.set_title(title)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Reward")
    ax.set_xticks(positions)
    ax.set_xticklabels(policy_labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return plot_path


def plot_reward_violin_by_policy_per_condition(
    plot_dir: Path,
    rewards: np.ndarray,
    policy_keys: np.ndarray,
    condition_keys: np.ndarray,
    *,
    clip_percentile_range: tuple[float, float] | None = (1.0, 99.0),
) -> list[Path]:
    """
    Create one reward violin plot per condition, grouped by policy.

    This makes it easier to compare which policy performs best within each
    condition.
    """

    condition_plot_dir = plot_dir / "reward_violin_by_condition"
    condition_plot_dir.mkdir(parents=True, exist_ok=True)

    rewards = np.asarray(rewards, dtype=np.float64)
    policy_keys = policy_keys.astype(str)
    condition_keys = condition_keys.astype(str)

    plot_paths = []

    for condition in sorted(np.unique(condition_keys)):

        condition_mask = condition_keys == condition

        condition_rewards = rewards[condition_mask]
        condition_policy_keys = policy_keys[condition_mask]

        policy_labels = []
        policy_reward_data = []

        for policy in sorted(np.unique(condition_policy_keys)):
            policy_mask = condition_policy_keys == policy

            policy_rewards = clip_reward_values(
                condition_rewards[policy_mask],
                clip_percentile_range=clip_percentile_range,
            )

            if len(policy_rewards) == 0:
                continue

            policy_labels.append(policy)
            policy_reward_data.append(policy_rewards)

        if len(policy_reward_data) == 0:
            continue

        safe_condition_name = str(condition).replace(" ", "_").replace("/", "_")

        plot_path = (
            condition_plot_dir
            / f"reward_violin_condition_{safe_condition_name}.png"
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        violin = ax.violinplot(
            policy_reward_data,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        colour_map = plt.get_cmap("tab10")

        for index, body in enumerate(violin["bodies"]):
            body.set_facecolor(colour_map(index % 10))
            body.set_edgecolor("black")
            body.set_alpha(0.8)

        positions = np.arange(1, len(policy_reward_data) + 1)

        q1_values = [np.percentile(values, 25) for values in policy_reward_data]
        median_values = [np.percentile(values, 50) for values in policy_reward_data]
        q3_values = [np.percentile(values, 75) for values in policy_reward_data]
        p5_values = [np.percentile(values, 5) for values in policy_reward_data]
        p95_values = [np.percentile(values, 95) for values in policy_reward_data]

        ax.vlines(
            positions,
            p5_values,
            p95_values,
            color="black",
            linewidth=1.2,
            alpha=0.7,
        )

        ax.vlines(
            positions,
            q1_values,
            q3_values,
            color="black",
            linewidth=5.0,
            label="Q1 to Q3",
        )

        ax.scatter(
            positions,
            median_values,
            color="white",
            edgecolor="black",
            zorder=3,
            label="Median",
        )

        y_min, y_max = ax.get_ylim()
        y_offset = (y_max - y_min) * 0.03

        for index, values in enumerate(policy_reward_data):
            ax.text(
                index + 1,
                min(np.max(values) + y_offset, y_max - y_offset),
                f"n={len(values)}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        title = f"Reward distribution by policy for condition: {condition}"

        if clip_percentile_range is not None:
            title += (
                f" ({clip_percentile_range[0]:.0f}-"
                f"{clip_percentile_range[1]:.0f} percentile clipped)"
            )

        ax.set_title(title)
        ax.set_xlabel("Policy")
        ax.set_ylabel("Reward")
        ax.set_xticks(positions)
        ax.set_xticklabels(policy_labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best", fontsize=9)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        plot_paths.append(plot_path)

    return plot_paths


def save_build_plots(
    save_dir: Path,
    extracted: dict,
    umap_embeddings: np.ndarray,
) -> list[Path]:
    """
    Generate useful plots for inspecting the built comparator database.
    """

    plot_dir = save_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    _, policy_keys, condition_names = get_plot_labels(extracted)

    saved_paths = []

    saved_paths.append(
        plot_overall_umap_by_policy_and_condition(
            plot_dir=plot_dir,
            umap_embeddings=umap_embeddings,
            policy_keys=policy_keys,
            condition_names=condition_names,
        )
    )

    saved_paths.extend(
        plot_policy_umap_by_condition(
            plot_dir=plot_dir,
            umap_embeddings=umap_embeddings,
            policy_keys=policy_keys,
            condition_names=condition_names,
        )
    )

    saved_paths.append(
        plot_reward_violin_by_policy(
            plot_dir=plot_dir,
            rewards=extracted["rewards"],
            policy_keys=policy_keys,
            clip_percentile_range=(1.0, 99.0),
        )
    )

    saved_paths.extend(
        plot_reward_violin_by_policy_per_condition(
            plot_dir=plot_dir,
            rewards=extracted["rewards"],
            policy_keys=policy_keys,
            condition_keys=condition_names,
            clip_percentile_range=(1.0, 99.0),
        )
    )

    return saved_paths

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
        export_parametric_umap_encoder_to_onnx(
            parametric_umap=parametric_umap,
            onnx_path=onnx_path,
            input_dim=embeddings.shape[1],
        )

        # Re-project the database embeddings through the exported ONNX encoder.
        # This guarantees that saved database UMAP coordinates use the exact same
        # projection path as runtime query embeddings.
        onnx_umap_embeddings = project_embeddings_with_umap_onnx(
            onnx_path=onnx_path,
            embeddings=embeddings,
        )

        check_parametric_umap_alignment(
            saved_umap_embeddings=umap_embeddings,
            onnx_umap_embeddings=onnx_umap_embeddings,
        )

        umap_embeddings = onnx_umap_embeddings

        umap_path = onnx_path
        runtime_umap_kind = "parametric_onnx"

    else:
        raise ValueError(f"Unsupported UMAP kind: {args.umap_kind}")

    # Save the comparator database
    database_path = save_comparator_database(save_dir=save_dir,
                                             extracted=extracted,
                                             umap_embeddings=umap_embeddings.astype(np.float32))

    # Generate build-time plots for inspecting the comparator database
    plot_paths = save_build_plots(
        save_dir=save_dir,
        extracted=extracted,
        umap_embeddings=umap_embeddings.astype(np.float32),
    )

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
        "sample_info": extracted["sample_info"],
        "gru_hyperparams": hp,
        "avg_embedding_gru_mse": float(extracted["avg_mse"]),
        "idx_to_terrain": extracted["idx_to_terrain"],
        "database_path": database_path.as_posix(),
        "embedding_gru_onnx_path": gru_onnx_path.as_posix(),
        "embedding_gru_stats_path": stats_path.as_posix(),
        "umap_path": umap_path.as_posix(),
        "plot_paths": [path.as_posix() for path in plot_paths],
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
    print(f"Plots: {save_dir / 'plots'}")
    print(f"YAML snippet: {config_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

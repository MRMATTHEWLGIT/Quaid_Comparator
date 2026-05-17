"""
Analyse and plot a comparator runtime run.

This script loads:
- runtime comparator SQLite logs from inference_times.db
- historical comparator_database.npz
- metadata.json from the comparator asset directory

It produces:
- enriched comparator_steps CSV
- UMAP plot with historical database embeddings and live query embeddings
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
import yaml

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Run directory helpers
# -------------------------------------------------------------------------

def resolve_run_paths(data_path: str | Path) -> dict[str, Path | None]:
    """
    Resolve all required paths from a saved comparator run directory.
    """

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Run data path does not exist: {data_path}")

    if not data_path.is_dir():
        raise NotADirectoryError(f"Run data path is not a directory: {data_path}")

    comparator_config_path = data_path / "comparator.yaml"
    env_config_path = data_path / "env.yaml"
    inference_db_path = data_path / "inference_times.db"

    if not comparator_config_path.exists():
        raise FileNotFoundError(
            f"Missing comparator config in run directory: {comparator_config_path}"
        )

    if not inference_db_path.exists():
        raise FileNotFoundError(
            f"Missing inference database in run directory: {inference_db_path}"
        )

    with open(comparator_config_path, "r", encoding="utf-8") as file:
        comparator_config = yaml.safe_load(file)

    comparator_assets_dir = Path(comparator_config["comparator_assets_dir"])

    database_path = comparator_assets_dir / "comparator_database.npz"
    metadata_path = comparator_assets_dir / "metadata.json"

    if not database_path.exists():
        raise FileNotFoundError(f"Missing comparator database: {database_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing comparator metadata: {metadata_path}")

    quaid_sqlite_files = sorted(data_path.glob("Quaid_*.sqlite"))
    quaid_sqlite_path = quaid_sqlite_files[0] if quaid_sqlite_files else None

    return {
        "data_path": data_path,
        "comparator_config_path": comparator_config_path,
        "env_config_path": env_config_path if env_config_path.exists() else None,
        "inference_db_path": inference_db_path,
        "quaid_sqlite_path": quaid_sqlite_path,
        "comparator_assets_dir": comparator_assets_dir,
        "database_path": database_path,
        "metadata_path": metadata_path,
    }

# -------------------------------------------------------------------------
# Loading helpers
# -------------------------------------------------------------------------

def load_comparator_steps(sqlite_path: str | Path) -> pd.DataFrame:
    """
    Load per-step comparator logs from inference_times.db.
    """

    sqlite_path = Path(sqlite_path)

    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite file does not exist: {sqlite_path}")

    conn = sqlite3.connect(sqlite_path)

    try:
        df = pd.read_sql_query(
            "SELECT * FROM comparator_steps ORDER BY episode_no, step",
            conn,
        )

    finally:
        conn.close()

    return df


def load_comparator_database(database_path: str | Path) -> dict:
    """
    Load comparator_database.npz as a simple dictionary.
    """

    database_path = Path(database_path)

    if not database_path.exists():
        raise FileNotFoundError(f"Comparator database does not exist: {database_path}")

    database = np.load(database_path, allow_pickle=True)

    return {
        key: database[key]
        for key in database.files
    }


def load_metadata(metadata_path: str | Path) -> dict:
    """
    Load comparator metadata JSON.
    """

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_json_cell(value, default):
    """
    Safely parse a JSON value stored in a SQLite text column.
    """

    if value is None:
        return default

    if isinstance(value, float) and np.isnan(value):
        return default

    if isinstance(value, (dict, list)):
        return value

    value = str(value).strip()

    if value == "":
        return default

    try:
        return json.loads(value)

    except json.JSONDecodeError:
        return default


# -------------------------------------------------------------------------
# Label helpers
# -------------------------------------------------------------------------

def terrain_name_from_id(terrain_id: int, idx_to_terrain: dict) -> str:
    """
    Convert a terrain ID into its saved terrain name.
    """

    terrain_key = str(int(terrain_id))

    if terrain_key not in idx_to_terrain:
        raise KeyError(f"Terrain ID {terrain_id} was not found in idx_to_terrain.")

    return str(idx_to_terrain[terrain_key])


def policy_key_from_terrain_name(terrain_name: str) -> str:
    """
    Extract the runtime policy key from a terrain name.
    """

    terrain_name = str(terrain_name).lower().strip()
    policy_part = terrain_name.split("+", maxsplit=1)[0]

    if policy_part in {"flat", "ramp", "uneven"}:
        return policy_part

    raise ValueError(f"Could not determine policy key from terrain name: {terrain_name}")


def add_database_labels(database: dict, metadata: dict) -> dict:
    """
    Add terrain names and policy keys to the loaded comparator database.
    """

    idx_to_terrain = metadata["idx_to_terrain"]

    terrain_names = np.array(
        [
            terrain_name_from_id(terrain_id, idx_to_terrain)
            for terrain_id in database["terrains"]
        ],
        dtype=object,
    )

    policy_keys = np.array(
        [
            policy_key_from_terrain_name(terrain_name)
            for terrain_name in terrain_names
        ],
        dtype=object,
    )

    database["terrain_names"] = terrain_names
    database["policy_keys"] = policy_keys

    return database


def add_vote_count_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand policy_vote_counts_json into one numeric column per policy.
    """

    df = df.copy()

    if "policy_vote_counts_json" not in df.columns:
        return df

    parsed_counts = df["policy_vote_counts_json"].apply(
        lambda value: parse_json_cell(value, default={})
    )

    all_policies = sorted(
        {
            str(policy)
            for counts in parsed_counts
            if isinstance(counts, dict)
            for policy in counts.keys()
        }
    )

    for policy in all_policies:
        column_name = f"vote_count_{policy}"

        df[column_name] = parsed_counts.apply(
            lambda counts, policy=policy: int(counts.get(policy, 0))
            if isinstance(counts, dict)
            else 0
        )

    return df


def add_candidate_filter_count_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand candidate_filter_counts_json into one numeric column per filter stage.
    """

    df = df.copy()

    if "candidate_filter_counts_json" not in df.columns:
        return df

    parsed_counts = df["candidate_filter_counts_json"].apply(
        lambda value: parse_json_cell(value, default={})
    )

    all_filter_keys = sorted(
        {
            str(key)
            for counts in parsed_counts
            if isinstance(counts, dict)
            for key in counts.keys()
        }
    )

    for key in all_filter_keys:
        column_name = f"filter_count_{key}"

        df[column_name] = parsed_counts.apply(
            lambda counts, key=key: int(counts.get(key, 0))
            if isinstance(counts, dict)
            else 0
        )

    return df


def add_candidate_umap_summary_columns(
    df: pd.DataFrame,
    database: dict,
) -> pd.DataFrame:
    """
    Add candidate-set UMAP summary columns using candidate_indices_json.

    Since voting no longer has one best candidate, this function stores the
    centroid of all valid candidates and the centroid of candidates belonging
    to the selected/next policy.
    """

    df = df.copy()

    if "candidate_indices_json" not in df.columns:
        return df

    umap_embeddings = database["umap_embeddings"]
    database_policy_keys = database["policy_keys"].astype(str)

    summary_columns = [
        "candidate_umap_centroid_x",
        "candidate_umap_centroid_y",
        "selected_policy_candidate_umap_centroid_x",
        "selected_policy_candidate_umap_centroid_y",
    ]

    for column in summary_columns:
        df[column] = np.nan

    for row_index, row in df.iterrows():

        candidate_indices = parse_json_cell(
            row.get("candidate_indices_json"),
            default=[],
        )

        if not isinstance(candidate_indices, list) or len(candidate_indices) == 0:
            continue

        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)

        valid_index_mask = (
            (candidate_indices >= 0)
            & (candidate_indices < len(umap_embeddings))
        )

        candidate_indices = candidate_indices[valid_index_mask]

        if len(candidate_indices) == 0:
            continue

        candidate_points = umap_embeddings[candidate_indices]

        df.at[row_index, "candidate_umap_centroid_x"] = float(
            np.mean(candidate_points[:, 0])
        )
        df.at[row_index, "candidate_umap_centroid_y"] = float(
            np.mean(candidate_points[:, 1])
        )

        selected_policy = str(row.get("next_policy"))
        selected_policy_mask = database_policy_keys[candidate_indices] == selected_policy

        if not np.any(selected_policy_mask):
            continue

        selected_points = candidate_points[selected_policy_mask]

        df.at[row_index, "selected_policy_candidate_umap_centroid_x"] = float(
            np.mean(selected_points[:, 0])
        )
        df.at[row_index, "selected_policy_candidate_umap_centroid_y"] = float(
            np.mean(selected_points[:, 1])
        )

    return df


def enrich_voting_steps(
    steps_df: pd.DataFrame,
    database: dict,
) -> pd.DataFrame:
    """
    Add voting-specific analysis columns to comparator_steps.
    """

    steps_df = steps_df.copy()

    steps_df = add_vote_count_columns(steps_df)
    steps_df = add_candidate_filter_count_columns(steps_df)
    steps_df = add_candidate_umap_summary_columns(
        df=steps_df,
        database=database,
    )

    if {"selected_policy_count", "candidate_count"}.issubset(steps_df.columns):
        candidate_count = steps_df["candidate_count"].replace(0, np.nan)
        steps_df["selected_policy_fraction_recomputed"] = (
            steps_df["selected_policy_count"] / candidate_count
        )

    return steps_df

# -------------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------------

def plot_umap_database_and_queries(
    database: dict,
    steps_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot historical database UMAP embeddings and overlay live query embeddings.

    The historical database is shown in the background. Query embeddings are
    shown above it and coloured by the policy active when the query was made.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    umap_embeddings = database["umap_embeddings"]
    database_policy_keys = database["policy_keys"]

    policy_markers = {
        "flat": "o",
        "ramp": "^",
        "uneven": "s",
    }

    policy_labels = ["flat", "ramp", "uneven"]

    fig, ax = plt.subplots(figsize=(10, 8))

    # ------------------------------------------------------------------
    # Plot database embeddings in background, separated by policy
    # ------------------------------------------------------------------

    for policy in policy_labels:

        mask = database_policy_keys == policy

        ax.scatter(
            umap_embeddings[mask, 0],
            umap_embeddings[mask, 1],
            s=12,
            alpha=0.18,
            marker=policy_markers.get(policy, "o"),
            label=f"database {policy}",
        )

    # ------------------------------------------------------------------
    # Plot live query embeddings on top, separated by current policy
    # ------------------------------------------------------------------

    query_df = steps_df.dropna(subset=["query_umap_x", "query_umap_y"]).copy()

    for policy in policy_labels:

        mask = query_df["current_policy"].astype(str) == policy

        if not mask.any():
            continue

        ax.scatter(
            query_df.loc[mask, "query_umap_x"],
            query_df.loc[mask, "query_umap_y"],
            s=42,
            alpha=0.95,
            marker=policy_markers.get(policy, "o"),
            edgecolors="black",
            linewidths=0.5,
            label=f"query {policy}",
        )

    # ------------------------------------------------------------------
    # Highlight committed switch points
    # ------------------------------------------------------------------

    if "switch_committed" in query_df.columns:

        switch_df = query_df[query_df["switch_committed"].astype(int) == 1]

        if len(switch_df) > 0:
            ax.scatter(
                switch_df["query_umap_x"],
                switch_df["query_umap_y"],
                s=150,
                facecolors="none",
                edgecolors="red",
                linewidths=2.4,
                label="committed switch",
                zorder=6,
            )

    episode_no = int(query_df["episode_no"].iloc[0])
    ax.set_title(
        f"Comparator UMAP Space: Historical Embeddings and Live Query Path - Episode {episode_no}"
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_query_trajectory(
    database: dict,
    steps_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot the live query trajectory through UMAP space with historical database
    embeddings shown lightly in the background.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query_df = steps_df.dropna(subset=["query_umap_x", "query_umap_y"]).copy()

    if len(query_df) == 0:
        return

    umap_embeddings = database["umap_embeddings"]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Historical database background
    ax.scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        s=8,
        c="lightgrey",
        alpha=0.18,
        linewidths=0,
        label="historical database",
    )

    # Live query path
    ax.plot(
        query_df["query_umap_x"],
        query_df["query_umap_y"],
        linewidth=1.4,
        alpha=0.85,
        zorder=3,
    )

    scatter = ax.scatter(
        query_df["query_umap_x"],
        query_df["query_umap_y"],
        c=query_df["step"],
        s=38,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.3,
        zorder=4,
    )

    fig.colorbar(scatter, ax=ax, label="Step")

    episode_no = int(query_df["episode_no"].iloc[0])

    ax.set_title(f"Live Query UMAP Trajectory - Episode {episode_no}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_vote_counts_over_time(
    steps_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot policy vote counts over time for one episode.

    Vote lines are shown with normal opacity when can_switch is active, and
    low opacity when the comparator is blocked by the switching gate.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vote_columns = [
        column
        for column in steps_df.columns
        if column.startswith("vote_count_")
    ]

    if len(vote_columns) == 0:
        return

    episode_df = steps_df.sort_values("step").copy()

    # If older databases do not have can_switch, treat all points as active.
    if "can_switch" in episode_df.columns:
        can_switch_mask = episode_df["can_switch"].astype(bool).to_numpy()
    else:
        can_switch_mask = np.ones(len(episode_df), dtype=bool)

    fig, ax = plt.subplots(figsize=(11, 5))

    for column in vote_columns:
        policy = column.replace("vote_count_", "")

        x_values = episode_df["step"].to_numpy()
        y_values = episode_df[column].to_numpy(dtype=float)

        # Active section: normal opacity
        y_active = y_values.copy()
        y_active[~can_switch_mask] = np.nan

        ax.plot(
            x_values,
            y_active,
            linewidth=1.8,
            alpha=1.0,
            label=policy,
        )

        # Inactive section: transparent/faded
        y_inactive = y_values.copy()
        y_inactive[can_switch_mask] = np.nan

        ax.plot(
            x_values,
            y_inactive,
            linewidth=1.5,
            alpha=0.18,
            label="_nolegend_",
        )

    if "switch_committed" in episode_df.columns:
        switch_df = episode_df[episode_df["switch_committed"].astype(int) == 1]

        for _, row in switch_df.iterrows():
            ax.axvline(
                x=row["step"],
                linewidth=1.0,
                alpha=0.35,
            )

    episode_no = int(episode_df["episode_no"].iloc[0])

    ax.set_title(f"Comparator Policy Vote Counts - Episode {episode_no}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Candidate votes")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def minmax_scale_to_range(
    values: np.ndarray,
    *,
    output_min: float = 0.0,
    output_max: float = 100.0,
) -> np.ndarray:
    """
    Min-max scale values into a fixed output range.

    If all values are equal, return the midpoint of the output range.
    """

    values = np.asarray(values, dtype=float)

    valid_mask = np.isfinite(values)

    scaled = np.full_like(values, np.nan, dtype=float)

    if not np.any(valid_mask):
        return scaled

    valid_values = values[valid_mask]

    value_min = np.min(valid_values)
    value_max = np.max(valid_values)

    if np.isclose(value_min, value_max):
        scaled[valid_mask] = (output_min + output_max) / 2.0
        return scaled

    scaled[valid_mask] = (
        (valid_values - value_min)
        / (value_max - value_min)
        * (output_max - output_min)
        + output_min
    )

    return scaled


def add_policy_background_shading(
    ax,
    episode_df: pd.DataFrame,
    *,
    policy_column: str = "current_policy",
    alpha: float = 0.08,
) -> list[Patch]:
    """
    Add background shading showing which policy was active over time.

    Returns legend handles for the policy shading.
    """

    if policy_column not in episode_df.columns:
        return []

    if "step" not in episode_df.columns:
        return []

    policy_df = episode_df.sort_values("step").copy()

    if len(policy_df) == 0:
        return []

    steps = policy_df["step"].to_numpy(dtype=float)
    policies = policy_df[policy_column].astype(str).to_numpy()

    unique_policies = sorted(policy_df[policy_column].astype(str).unique())

    colour_map = plt.get_cmap("tab10")

    policy_colours = {
        policy: colour_map(index % 10)
        for index, policy in enumerate(unique_policies)
    }

    if len(steps) > 1:
        step_delta = float(np.median(np.diff(steps)))
    else:
        step_delta = 1.0

    # Identify contiguous policy segments.
    segment_start_index = 0

    for index in range(1, len(policy_df) + 1):

        reached_end = index == len(policy_df)

        policy_changed = (
            not reached_end
            and policies[index] != policies[segment_start_index]
        )

        if reached_end or policy_changed:

            policy = policies[segment_start_index]

            start_step = steps[segment_start_index] - 0.5 * step_delta
            end_step = steps[index - 1] + 0.5 * step_delta

            ax.axvspan(
                start_step,
                end_step,
                color=policy_colours[policy],
                alpha=alpha,
                linewidth=0,
                zorder=0,
            )

            segment_start_index = index

    legend_handles = [
        Patch(
            facecolor=policy_colours[policy],
            alpha=alpha,
            label=f"Active policy: {policy}",
        )
        for policy in unique_policies
    ]

    return legend_handles


def plot_episode_step_reward_scaled(
    steps_df: pd.DataFrame,
    output_path: str | Path,
    *,
    rolling_window: int = 50,
) -> None:
    """
    Plot raw and smoothed min-max scaled step reward for one episode.

    The raw signal is shown transparently, while the rolling mean is shown
    strongly. This makes local reward changes around switches easier to see.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {"episode_no", "step", "step_reward_total"}

    if not required_columns.issubset(steps_df.columns):
        return

    episode_df = steps_df.sort_values("step").copy()

    episode_no = int(episode_df["episode_no"].iloc[0])

    step_values = episode_df["step"].to_numpy()
    step_reward = episode_df["step_reward_total"].to_numpy(dtype=float)

    scaled_step_reward = minmax_scale_to_range(
        step_reward,
        output_min=0.0,
        output_max=100.0,
    )

    episode_df["step_reward_scaled"] = scaled_step_reward

    episode_df["step_reward_scaled_smooth"] = (
        episode_df["step_reward_scaled"]
        .rolling(window=rolling_window, min_periods=1, center=True)
        .mean()
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    # Raw scaled reward, very transparent
    ax.plot(
        episode_df["step"],
        episode_df["step_reward_scaled"],
        linewidth=0.7,
        alpha=0.15,
        label="Step reward scaled 0-100",
    )

    # Smoothed scaled reward, main signal
    ax.plot(
        episode_df["step"],
        episode_df["step_reward_scaled_smooth"],
        linewidth=2.0,
        alpha=0.95,
        label=f"Rolling mean, window={rolling_window}",
    )

    if "switch_committed" in episode_df.columns:
        switch_df = episode_df[episode_df["switch_committed"].astype(int) == 1]

        for _, row in switch_df.iterrows():
            ax.axvline(
                x=row["step"],
                linewidth=1.0,
                alpha=0.35,
            )

    ax.set_title(f"Comparator Smoothed Step Reward - Episode {episode_no}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Min-max scaled step reward")
    ax.set_ylim(-5.0, 105.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_episode_reward_rate(
    steps_df: pd.DataFrame,
    output_path: str | Path,
    *,
    rolling_window: int = 50,
) -> None:
    """
    Plot rolling reward rate over one episode.

    Background shading shows which policy was active at each timestep.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {"episode_no", "step", "step_reward_total"}

    if not required_columns.issubset(steps_df.columns):
        return

    episode_df = steps_df.sort_values("step").copy()

    episode_no = int(episode_df["episode_no"].iloc[0])

    episode_df["reward_rate_smooth"] = (
        episode_df["step_reward_total"]
        .rolling(window=rolling_window, min_periods=1, center=True)
        .mean()
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    policy_handles = add_policy_background_shading(
        ax,
        episode_df,
        policy_column="current_policy",
        alpha=0.08,
    )

    ax.plot(
        episode_df["step"],
        episode_df["reward_rate_smooth"],
        linewidth=2.0,
        label=f"Rolling reward rate, window={rolling_window}",
        zorder=3,
    )

    if "switch_committed" in episode_df.columns:
        switch_df = episode_df[episode_df["switch_committed"].astype(int) == 1]

        for _, row in switch_df.iterrows():
            ax.axvline(
                x=row["step"],
                linewidth=1.0,
                alpha=0.45,
                zorder=4,
            )

    ax.set_title(f"Comparator Rolling Reward Rate - Episode {episode_no}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Raw rolling step reward")
    ax.grid(True, alpha=0.25)

    line_handles, line_labels = ax.get_legend_handles_labels()

    ax.legend(
        handles=line_handles + policy_handles,
        loc="best",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_episode_reward_components(
    steps_df: pd.DataFrame,
    output_path: str | Path,
    *,
    rolling_window: int = 50,
) -> None:
    """
    Plot smoothed distance, roll, and current reward components.

    Background shading shows which policy was active at each timestep.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {
        "episode_no",
        "step",
        "reward_distance",
        "reward_roll",
        "reward_current",
        "reward_yaw",
        "reward_pitch",
        "reward_action_smoothness",
    }

    if not required_columns.issubset(steps_df.columns):
        return

    episode_df = steps_df.sort_values("step").copy()

    episode_no = int(episode_df["episode_no"].iloc[0])

    component_columns = [
        "reward_distance",
        "reward_roll",
        "reward_current",
        "reward_yaw",
        "reward_pitch",
        "reward_action_smoothness",
    ]

    fig, ax = plt.subplots(figsize=(11, 5))

    policy_handles = add_policy_background_shading(
        ax,
        episode_df,
        policy_column="current_policy",
        alpha=0.08,
    )

    for column in component_columns:
        smoothed_column = f"{column}_smooth"

        episode_df[smoothed_column] = (
            episode_df[column]
            .rolling(window=rolling_window, min_periods=1, center=True)
            .mean()
        )

        label = column.replace("reward_", "")

        ax.plot(
            episode_df["step"],
            episode_df[smoothed_column],
            linewidth=1.8,
            label=f"{label}, rolling mean",
            zorder=3,
        )

    if "switch_committed" in episode_df.columns:
        switch_df = episode_df[episode_df["switch_committed"].astype(int) == 1]

        for _, row in switch_df.iterrows():
            ax.axvline(
                x=row["step"],
                linewidth=1.0,
                alpha=0.45,
                zorder=4,
            )

    ax.set_title(f"Comparator Reward Components - Episode {episode_no}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Raw rolling reward component value")
    ax.grid(True, alpha=0.25)

    line_handles, line_labels = ax.get_legend_handles_labels()

    ax.legend(
        handles=line_handles + policy_handles,
        loc="best",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to a saved comparator run directory.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where analysis CSVs and plots should be saved. "
            "Defaults to <data-path>/analysis_results."
        ),
    )

    args = parser.parse_args()

    paths = resolve_run_paths(args.data_path)

    if args.output_dir is None:
        output_dir = paths["data_path"] / "analysis_results"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    steps_df = load_comparator_steps(paths["inference_db_path"])

    database = load_comparator_database(paths["database_path"])
    metadata = load_metadata(paths["metadata_path"])
    database = add_database_labels(database, metadata)

    steps_df = enrich_voting_steps(
        steps_df=steps_df,
        database=database,
    )

    # Save enriched CSV
    csv_path = output_dir / "comparator_steps_enriched.csv"
    steps_df.to_csv(csv_path, index=False)

    # Save per-episode plots
    episode_plot_dir = output_dir / "episodes"
    episode_plot_dir.mkdir(parents=True, exist_ok=True)

    saved_plot_paths = []

    for episode_no, episode_df in steps_df.groupby("episode_no"):

        database_query_plot_path = (
            episode_plot_dir / f"episode_{int(episode_no):03d}_umap_database_with_queries.png"
        )

        query_trajectory_plot_path = (
            episode_plot_dir / f"episode_{int(episode_no):03d}_query_umap_trajectory.png"
        )

        plot_umap_database_and_queries(
            database=database,
            steps_df=episode_df,
            output_path=database_query_plot_path,
        )

        plot_query_trajectory(
            database=database,
            steps_df=episode_df,
            output_path=query_trajectory_plot_path,
        )

        vote_counts_plot_path = (
            episode_plot_dir / f"episode_{int(episode_no):03d}_policy_vote_counts.png"
        )

        plot_vote_counts_over_time(
            steps_df=episode_df,
            output_path=vote_counts_plot_path,
        )

        plot_episode_step_reward_scaled(
            steps_df=episode_df,
            output_path=episode_plot_dir / f"episode_{int(episode_no):03d}_step_reward_scaled_smooth.png",
            rolling_window=50,
        )

        plot_episode_reward_rate(
            steps_df=episode_df,
            output_path=episode_plot_dir / f"episode_{int(episode_no):03d}_reward_rate.png",
            rolling_window=50,
        )

        plot_episode_reward_components(
            steps_df=episode_df,
            output_path=episode_plot_dir / f"episode_{int(episode_no):03d}_reward_components.png",
            rolling_window=50,
        )

        saved_plot_paths.extend([
            database_query_plot_path,
            query_trajectory_plot_path,
        ])

    print(f"Run directory:       {paths['data_path']}")
    print(f"Comparator config:   {paths['comparator_config_path']}")
    print(f"Environment config:  {paths['env_config_path']}")
    print(f"Inference database:  {paths['inference_db_path']}")
    print(f"Quaid SQLite:        {paths['quaid_sqlite_path']}")
    print(f"Comparator assets:   {paths['comparator_assets_dir']}")
    print()
    print(f"Loaded {len(steps_df)} comparator steps.")
    print(f"Saved CSV:           {csv_path}")
    print(f"Saved episode plots: {episode_plot_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
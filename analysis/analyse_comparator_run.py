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
    

def normalise_runtime_step_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise runtime logging columns so analysis works for both old and new
    comparator logs.

    New logs contain one row per robot step. Only some rows have
    comparator_ran == 1.
    """

    df = df.copy()

    numeric_columns = [
        "episode_no",
        "step",
        "switch_committed",
        "can_switch",
        "candidate_count",
        "selected_policy_count",
        "selected_policy_fraction",
        "second_policy_count",
        "second_policy_fraction",
        "vote_margin",
        "query_local_reward_mean",
        "query_umap_x",
        "query_umap_y",
        "reward_distance",
        "reward_roll",
        "reward_current",
        "reward_yaw",
        "reward_pitch",
        "reward_action_smoothness",
        "step_reward_total",
        "episode_reward_total",
        "comparator_ran",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Backwards compatibility for older logs that do not have comparator_ran.
    # In old logs, every stored row was effectively a comparator row.
    if "comparator_ran" not in df.columns:
        df["comparator_ran"] = 1

    df["comparator_ran"] = (
        pd.to_numeric(df["comparator_ran"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    if "switch_committed" in df.columns:
        df["switch_committed"] = (
            pd.to_numeric(df["switch_committed"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    if "can_switch" in df.columns:
        df["can_switch"] = (
            pd.to_numeric(df["can_switch"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    return df


def get_comparator_ran_mask(df: pd.DataFrame) -> pd.Series:
    """
    Return True only for rows where the expensive comparator selection actually ran.
    """

    if df.empty:
        return pd.Series(False, index=df.index)

    if "comparator_ran" not in df.columns:
        # Backwards compatibility for older logs.
        return pd.Series(True, index=df.index)

    return (
        pd.to_numeric(df["comparator_ran"], errors="coerce")
        .fillna(0)
        .astype(int)
        == 1
    )


def get_comparator_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows where comparator selection ran.
    """

    if df.empty:
        return df.copy()

    return df[get_comparator_ran_mask(df)].copy()


def get_query_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only fresh comparator rows with valid UMAP query coordinates.
    """

    required_columns = {"query_umap_x", "query_umap_y"}

    if df.empty or not required_columns.issubset(df.columns):
        return pd.DataFrame()

    comparator_df = get_comparator_rows(df)

    return comparator_df.dropna(subset=["query_umap_x", "query_umap_y"]).copy()


def add_playbook_columns(
    steps_df: pd.DataFrame,
    comparator_config_path: str | Path,
) -> pd.DataFrame:
    """
    Add playbook_entry and episode_uses_comparator columns using comparator.yaml.
    """

    steps_df = steps_df.copy()

    with open(comparator_config_path, "r", encoding="utf-8") as file:
        comparator_config = yaml.safe_load(file)

    episode_playbook = comparator_config.get("episode_playbook", [])

    def get_playbook_entry(episode_no):
        episode_index = int(episode_no)

        if 0 <= episode_index < len(episode_playbook):
            return str(episode_playbook[episode_index])

        return "unknown"

    steps_df["playbook_entry"] = steps_df["episode_no"].apply(get_playbook_entry)
    steps_df["episode_uses_comparator"] = (
        steps_df["playbook_entry"].astype(str) == "comparator"
    )

    return steps_df


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

    comparator_rows = get_comparator_rows(df)

    for row_index, row in comparator_rows.iterrows():

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

    query_df = get_query_rows(steps_df)

    if len(query_df) == 0:
        plt.close(fig)
        return

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

    query_df = get_query_rows(steps_df)

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

    episode_df = get_comparator_rows(steps_df).sort_values("step").copy()

    if len(episode_df) == 0:
        return

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


def plot_switch_reward_delta(
    steps_df: pd.DataFrame,
    output_path: str | Path,
    *,
    window: int = 50,
) -> None:
    """
    Plot mean reward before and after each committed policy switch.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {
        "episode_no",
        "step",
        "step_reward_total",
        "switch_committed",
        "current_policy",
        "next_policy",
    }

    if not required_columns.issubset(steps_df.columns):
        return

    episode_df = steps_df.sort_values("step").copy()
    episode_no = int(episode_df["episode_no"].iloc[0])

    switch_df = episode_df[episode_df["switch_committed"].astype(int) == 1]

    if len(switch_df) == 0:
        return

    labels = []
    before_means = []
    after_means = []
    deltas = []

    for _, switch_row in switch_df.iterrows():

        switch_step = int(switch_row["step"])

        before_df = episode_df[
            (episode_df["step"] >= switch_step - window)
            & (episode_df["step"] < switch_step)
        ]

        after_df = episode_df[
            (episode_df["step"] > switch_step)
            & (episode_df["step"] <= switch_step + window)
        ]

        if len(before_df) == 0 or len(after_df) == 0:
            continue

        before_mean = float(before_df["step_reward_total"].mean())
        after_mean = float(after_df["step_reward_total"].mean())

        before_policy = str(before_df["current_policy"].iloc[-1])
        after_policy = str(after_df["current_policy"].iloc[0])

        labels.append(f"{switch_step}\n{before_policy}→{after_policy}")
        before_means.append(before_mean)
        after_means.append(after_mean)
        deltas.append(after_mean - before_mean)

    if len(labels) == 0:
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        x - width / 2,
        before_means,
        width,
        label=f"Before switch, {window} steps",
    )

    ax.bar(
        x + width / 2,
        after_means,
        width,
        label=f"After switch, {window} steps",
    )

    for index, delta in enumerate(deltas):
        ax.text(
            x[index],
            max(before_means[index], after_means[index]),
            f"Δ={delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(f"Switch Reward Change - Episode {episode_no}")
    ax.set_xlabel("Switch step and policy transition")
    ax.set_ylabel("Mean step reward")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def infer_episode_label(episode_df: pd.DataFrame) -> str:
    """
    Infer whether an episode was fixed-policy or comparator-controlled.
    """

    if "playbook_entry" in episode_df.columns:
        playbook_entry = str(episode_df["playbook_entry"].iloc[0])
        if playbook_entry:
            return playbook_entry

    if "episode_uses_comparator" in episode_df.columns:
        uses_comparator = bool(episode_df["episode_uses_comparator"].astype(bool).any())
        if uses_comparator:
            return "comparator"

    if "can_switch" in episode_df.columns and episode_df["can_switch"].astype(bool).any():
        return "comparator"

    first_policy = str(episode_df["current_policy"].iloc[0])
    return first_policy


def plot_final_reward_per_episode(steps_df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Plot final episode reward for each episode.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for episode_no, episode_df in steps_df.groupby("episode_no"):

        episode_df = episode_df.sort_values("step").copy()

        if "episode_reward_total" in episode_df.columns:
            final_reward = float(episode_df["episode_reward_total"].iloc[-1])
        else:
            final_reward = float(episode_df["step_reward_total"].sum())

        records.append({
            "episode_no": int(episode_no),
            "episode_label": infer_episode_label(episode_df),
            "final_reward": final_reward,
        })

    summary_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(summary_df))

    ax.bar(x, summary_df["final_reward"])

    ax.set_title("Final Cumulative Reward Per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Final cumulative reward")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"Ep {row.episode_no}\n{row.episode_label}"
            for row in summary_df.itertuples()
        ]
    )

    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cumulative_reward_curves(steps_df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Plot cumulative reward curves for all episodes on one figure.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))

    for episode_no, episode_df in steps_df.groupby("episode_no"):

        episode_df = episode_df.sort_values("step").copy()
        episode_label = infer_episode_label(episode_df)

        if "episode_reward_total" in episode_df.columns:
            cumulative_reward = episode_df["episode_reward_total"]
        else:
            cumulative_reward = episode_df["step_reward_total"].cumsum()

        ax.plot(
            episode_df["step"],
            cumulative_reward,
            linewidth=1.8,
            label=f"Ep {int(episode_no)} ({episode_label})",
        )

    ax.set_title("Cumulative Reward Curves")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative reward")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_policy_occupancy_per_episode(
    steps_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot policy occupancy fraction for each episode as stacked bars.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    policies = sorted(steps_df["current_policy"].astype(str).unique())

    records = []

    for episode_no, episode_df in steps_df.groupby("episode_no"):

        episode_df = episode_df.copy()
        total_steps = len(episode_df)

        record = {
            "episode_no": int(episode_no),
            "episode_label": infer_episode_label(episode_df),
        }

        for policy in policies:
            record[policy] = float(
                (episode_df["current_policy"].astype(str) == policy).sum()
                / total_steps
            )

        records.append(record)

    occupancy_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(11, 5))

    x = np.arange(len(occupancy_df))
    bottom = np.zeros(len(occupancy_df))

    for policy in policies:

        values = occupancy_df[policy].to_numpy(dtype=float)

        ax.bar(
            x,
            values,
            bottom=bottom,
            label=policy,
        )

        bottom += values

    ax.set_title("Policy Occupancy Per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Fraction of timesteps")
    ax.set_ylim(0.0, 1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"Ep {row.episode_no}\n{row.episode_label}"
            for row in occupancy_df.itertuples()
        ]
    )

    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_policy_timeline(
    steps_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot active policy timeline for one episode.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episode_df = steps_df.sort_values("step").copy()

    policies = sorted(episode_df["current_policy"].astype(str).unique())
    policy_to_y = {
        policy: index
        for index, policy in enumerate(policies)
    }

    y_values = episode_df["current_policy"].astype(str).map(policy_to_y)

    episode_no = int(episode_df["episode_no"].iloc[0])
    episode_label = infer_episode_label(episode_df)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        episode_df["step"],
        y_values,
        linewidth=2.0,
    )

    if "switch_committed" in episode_df.columns:

        switch_df = episode_df[episode_df["switch_committed"].astype(int) == 1]

        if len(switch_df) > 0:
            switch_y = switch_df["current_policy"].astype(str).map(policy_to_y)

            ax.scatter(
                switch_df["step"],
                switch_y,
                marker="x",
                s=120,
                linewidths=2.5,
                label="Committed switch",
                zorder=5,
            )

    ax.set_title(f"Policy Timeline - Episode {episode_no} ({episode_label})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Active policy")

    ax.set_yticks(list(policy_to_y.values()))
    ax.set_yticklabels(list(policy_to_y.keys()))

    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()

    if len(handles) > 0:
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_aggregate_reward_rate(
    steps_df: pd.DataFrame,
    output_path: str | Path,
    *,
    rolling_window: int = 50,
) -> None:
    """
    Plot rolling reward rate for all episodes on one aggregate plot.

    Each episode is shown as one line. Comparator switch events are marked
    with an 'x' marker on the corresponding episode line.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {
        "episode_no",
        "step",
        "step_reward_total",
    }

    if not required_columns.issubset(steps_df.columns):
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for episode_no, episode_df in steps_df.groupby("episode_no"):

        episode_df = episode_df.sort_values("step").copy()

        episode_df["reward_rate_smooth"] = (
            episode_df["step_reward_total"]
            .rolling(
                window=rolling_window,
                min_periods=1,
                center=True,
            )
            .mean()
        )

        episode_label = infer_episode_label(episode_df)

        is_comparator_episode = (str(episode_label).strip().lower() == "comparator")

        line, = ax.plot(
            episode_df["step"],
            episode_df["reward_rate_smooth"],
            label=f"Ep {int(episode_no)} ({episode_label})",
            zorder=4 if is_comparator_episode else 3,
        )

        if is_comparator_episode:
            line.set_linestyle("-")
            line.set_alpha(1.0)
            line.set_linewidth(2.4)
        else:
            line.set_linestyle("-")
            line.set_alpha(0.55)
            line.set_linewidth(1.8)


        if "switch_committed" in episode_df.columns:

            switch_df = episode_df[
                episode_df["switch_committed"].astype(int) == 1
            ]

            if len(switch_df) > 0:

                ax.scatter(
                    switch_df["step"],
                    switch_df["reward_rate_smooth"],
                    marker="X",
                    s=180,
                    facecolors=line.get_color(),
                    edgecolors="black",
                    linewidths=1.8,
                    zorder=6,
                )

    ax.set_title(
        f"Aggregate Rolling Reward Rate Across Episodes "
        f"(window={rolling_window})"
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Raw rolling step reward")

    ax.grid(True, alpha=0.25)

    # Add one dummy marker so the legend explains the switch symbols.
    ax.scatter(
        [],
        [],
        marker="X",
        s=180,
        facecolors="white",
        edgecolors="black",
        linewidths=1.8,
        label="Committed comparator switch",
    )

    ax.legend(loc="best", fontsize=9)

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
    steps_df = normalise_runtime_step_columns(steps_df)
    steps_df = add_playbook_columns(
        steps_df=steps_df,
        comparator_config_path=paths["comparator_config_path"],
    )

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

    # Save aggregate multi-episode plots
    aggregate_plot_dir = output_dir / "aggregate"
    aggregate_plot_dir.mkdir(parents=True, exist_ok=True)

    plot_final_reward_per_episode(
        steps_df=steps_df,
        output_path=aggregate_plot_dir / "final_reward_per_episode.png",
    )

    plot_cumulative_reward_curves(
        steps_df=steps_df,
        output_path=aggregate_plot_dir / "cumulative_reward_curves.png",
    )

    plot_policy_occupancy_per_episode(
        steps_df=steps_df,
        output_path=aggregate_plot_dir / "policy_occupancy_per_episode.png",
    )

    plot_aggregate_reward_rate(
        steps_df=steps_df,
        output_path=aggregate_plot_dir / "aggregate_reward_rate.png",
        rolling_window=50,
    )

    # Save per-episode plots
    episode_root_dir = output_dir / "episodes"
    episode_root_dir.mkdir(parents=True, exist_ok=True)

    saved_plot_paths = []

    for episode_no, episode_df in steps_df.groupby("episode_no"):

        episode_no_int = int(episode_no)

        # Create a separate folder for each episode
        episode_plot_dir = episode_root_dir / f"episode_{episode_no_int:03d}"
        episode_plot_dir.mkdir(parents=True, exist_ok=True)

        database_query_plot_path = (
            episode_plot_dir / "umap_database_with_queries.png"
        )

        query_trajectory_plot_path = (
            episode_plot_dir / "query_umap_trajectory.png"
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
            episode_plot_dir / "policy_vote_counts.png"
        )
        
        plot_policy_timeline(
            steps_df=episode_df,
            output_path=episode_plot_dir / "policy_timeline.png",
        )

        plot_vote_counts_over_time(
            steps_df=episode_df,
            output_path=vote_counts_plot_path,
        )

        plot_switch_reward_delta(
            steps_df=episode_df,
            output_path=episode_plot_dir / "switch_reward_delta.png",
            window=50,
        )

        plot_episode_reward_rate(
            steps_df=episode_df,
            output_path=episode_plot_dir / "reward_rate.png",
            rolling_window=50,
        )

        plot_episode_reward_components(
            steps_df=episode_df,
            output_path=episode_plot_dir / "reward_components.png",
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
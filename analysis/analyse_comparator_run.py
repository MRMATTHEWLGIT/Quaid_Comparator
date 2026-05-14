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


# -------------------------------------------------------------------------
# Enrichment helpers
# -------------------------------------------------------------------------

def add_best_candidate_umap_coordinates(
    df: pd.DataFrame,
    database: dict,
) -> pd.DataFrame:
    """
    Add best-candidate UMAP coordinates using best_idx.
    """

    df = df.copy()

    umap_embeddings = database["umap_embeddings"]

    df["best_candidate_umap_x"] = np.nan
    df["best_candidate_umap_y"] = np.nan

    valid = df["best_idx"].notna()

    best_indices = df.loc[valid, "best_idx"].astype(int).to_numpy()

    df.loc[valid, "best_candidate_umap_x"] = umap_embeddings[best_indices, 0]
    df.loc[valid, "best_candidate_umap_y"] = umap_embeddings[best_indices, 1]

    return df


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
                s=95,
                facecolors="none",
                edgecolors="black",
                linewidths=1.6,
                label="committed switch",
            )

    ax.set_title("Comparator UMAP Space: Database Embeddings and Live Query Path")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_query_trajectory(
    steps_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot only the live query trajectory through UMAP space.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query_df = steps_df.dropna(subset=["query_umap_x", "query_umap_y"]).copy()

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(
        query_df["query_umap_x"],
        query_df["query_umap_y"],
        linewidth=1.2,
        alpha=0.8,
    )

    scatter = ax.scatter(
        query_df["query_umap_x"],
        query_df["query_umap_y"],
        c=query_df["step"],
        s=35,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.3,
    )

    fig.colorbar(scatter, ax=ax, label="Step")

    ax.set_title("Live Query UMAP Trajectory")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.25)

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

    steps_df = add_best_candidate_umap_coordinates(
        df=steps_df,
        database=database,
    )

    # Save enriched CSV
    csv_path = output_dir / "comparator_steps_enriched.csv"
    steps_df.to_csv(csv_path, index=False)

    # Save plots
    database_query_plot_path = output_dir / "umap_database_with_queries.png"
    query_trajectory_plot_path = output_dir / "query_umap_trajectory.png"

    plot_umap_database_and_queries(
        database=database,
        steps_df=steps_df,
        output_path=database_query_plot_path,
    )

    plot_query_trajectory(
        steps_df=steps_df,
        output_path=query_trajectory_plot_path,
    )

    print(f"Run directory:       {paths['data_path']}")
    print(f"Comparator config:   {paths['comparator_config_path']}")
    print(f"Environment config:  {paths['env_config_path']}")
    print(f"Inference database:  {paths['inference_db_path']}")
    print(f"Quaid SQLite:        {paths['quaid_sqlite_path']}")
    print(f"Comparator assets:   {paths['comparator_assets_dir']}")
    print()
    print(f"Loaded {len(steps_df)} comparator steps.")
    print(f"Saved CSV:           {csv_path}")
    print(f"Saved plot:          {database_query_plot_path}")
    print(f"Saved plot:          {query_trajectory_plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
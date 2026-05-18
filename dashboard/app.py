"""Live Streamlit dashboard for comparator MQTT telemetry."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative

from dashboard.mqtt_state import LiveTelemetryState


DEFAULT_TOPIC = "quaid/comparator/r0/telemetry"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live comparator dashboard")

    parser.add_argument(
        "--comparator-path",
        default=None,
        help="Path to the comparator asset folder containing comparator_database.npz, "
            "metadata.json, embedding_gru_encoder.onnx, and parametric_umap_encoder.onnx.",
    )

    parser.add_argument(
        "--mqtt-host",
        default="localhost",
        help="MQTT broker host used by the comparator telemetry publisher.",
    )

    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=1883,
        help="MQTT broker port.",
    )

    parser.add_argument(
        "--mqtt-queue",
        type=int,
        required=True,
        help="MQTT queue number. The dashboard subscribes to "
            "quaid/comparator/r<mqtt-queue>/telemetry.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum live steps to retain for the current episode.",
    )

    parser.add_argument(
        "--rolling-window",
        type=int,
        default=50,
        help="Rolling window for the reward-rate plot.",
    )

    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=500,
        help="Dashboard auto-refresh interval in milliseconds.",
    )

    args, _ = parser.parse_known_args(sys.argv[1:])

    return args


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_live_state(
    mqtt_host: str,
    mqtt_port: int,
    mqtt_topic: str,
    max_steps: int,
) -> LiveTelemetryState:
    return LiveTelemetryState(
        host=mqtt_host,
        port=mqtt_port,
        topic=mqtt_topic,
        max_steps=max_steps,
    )


@st.cache_data(show_spinner=True)
def load_comparator_database(comparator_path_string: str) -> dict[str, Any]:
    comparator_path, database_path = resolve_comparator_database_path(comparator_path_string)

    npz = np.load(database_path, allow_pickle=True)
    database = {key: npz[key] for key in npz.files}

    umap_embeddings = np.asarray(database["umap_embeddings"], dtype=float)
    n_points = len(umap_embeddings)

    policy_keys = derive_policy_keys(database, database_path, n_points)
    condition_names = derive_condition_names(database, database_path, n_points)

    return {
        "comparator_path": str(comparator_path),
        "path": str(database_path),
        "umap_embeddings": umap_embeddings,
        "policy_keys": policy_keys,
        "condition_names": condition_names,
    }


def resolve_comparator_database_path(comparator_path_string: str) -> tuple[Path, Path]:
    """
    Resolve a comparator asset folder into its comparator_database.npz path.

    Expected folder structure:

        comparator_path/
        ├── comparator_database.npz
        ├── metadata.json
        ├── embedding_gru_encoder.onnx
        ├── parametric_umap_encoder.onnx
        └── ...

    The input must be the comparator asset folder, not the database file itself.
    """

    comparator_path = Path(comparator_path_string)

    if not comparator_path.exists():
        raise FileNotFoundError(
            f"Comparator path does not exist: {comparator_path}"
        )

    if not comparator_path.is_dir():
        raise NotADirectoryError(
            "Expected --comparator-path to point to a comparator asset folder, "
            f"but got a file: {comparator_path}"
        )

    database_path = comparator_path / "comparator_database.npz"

    if not database_path.exists():
        raise FileNotFoundError(
            "Could not find comparator_database.npz inside comparator path: "
            f"{comparator_path}"
        )

    return comparator_path, database_path


# ---------------------------------------------------------------------------
# Database label helpers
# ---------------------------------------------------------------------------


def derive_policy_keys(
    database: dict[str, Any],
    database_path: Path,
    n_points: int,
) -> np.ndarray:
    if "policy_keys" in database:
        return np.asarray(database["policy_keys"]).astype(str)

    terrain_names = derive_terrain_names(database, database_path)

    if terrain_names is not None:
        return np.asarray(
            [str(name).lower().strip().split("+", maxsplit=1)[0] for name in terrain_names],
            dtype=object,
        )

    return np.asarray(["database"] * n_points, dtype=object)


def derive_condition_names(
    database: dict[str, Any],
    database_path: Path,
    n_points: int,
) -> np.ndarray:
    if "condition_names" in database:
        return np.asarray(database["condition_names"]).astype(str)

    terrain_names = derive_terrain_names(database, database_path)

    if terrain_names is not None:
        condition_names = []

        for terrain_name in terrain_names:
            terrain_name = str(terrain_name).lower().strip()
            condition_names.append(terrain_name.rsplit("-", maxsplit=1)[-1])

        return np.asarray(condition_names, dtype=object)

    return np.asarray(["unknown"] * n_points, dtype=object)


def derive_terrain_names(
    database: dict[str, Any],
    database_path: Path,
) -> np.ndarray | None:
    if "terrain_names" in database:
        return np.asarray(database["terrain_names"]).astype(str)

    if "terrains" not in database:
        return None

    metadata_path = database_path.parent / "metadata.json"

    if not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    idx_to_terrain = metadata.get("idx_to_terrain")

    if not isinstance(idx_to_terrain, dict):
        return None

    terrains = np.asarray(database["terrains"])

    return np.asarray(
        [str(idx_to_terrain[str(int(terrain_id))]) for terrain_id in terrains],
        dtype=object,
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def get_query_df(steps_df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with valid live query UMAP coordinates."""

    required_columns = {"query_umap_x", "query_umap_y"}

    if steps_df.empty or not required_columns.issubset(steps_df.columns):
        return pd.DataFrame()

    return steps_df.dropna(subset=["query_umap_x", "query_umap_y"]).copy()


def get_switch_mask(df: pd.DataFrame) -> pd.Series:
    """Return a safe boolean mask for committed-switch rows."""

    if df.empty or "switch_committed" not in df.columns:
        return pd.Series(False, index=df.index)

    return (
        pd.to_numeric(df["switch_committed"], errors="coerce")
        .fillna(0)
        .astype(int)
        == 1
    )


def make_reward_rate_figure(steps_df: pd.DataFrame, rolling_window: int) -> go.Figure:
    fig = go.Figure()

    if steps_df.empty or "step_reward_total" not in steps_df:
        fig.update_layout(title="Rolling reward rate")
        return fig

    episode_df = steps_df.sort_values("step").copy()
    episode_df["reward_rate_smooth"] = (
        episode_df["step_reward_total"]
        .astype(float)
        .rolling(window=rolling_window, min_periods=1, center=True)
        .mean()
    )

    add_policy_background_vrects(fig, episode_df, policy_column="current_policy")

    fig.add_trace(
        go.Scatter(
            x=episode_df["step"],
            y=episode_df["reward_rate_smooth"],
            mode="lines",
            line={"width": 3},
            name=f"Rolling reward, window={rolling_window}",
        )
    )

    switch_df = episode_df[get_switch_mask(episode_df)]

    for _, row in switch_df.iterrows():
        fig.add_vline(
            x=float(row["step"]),
            line_width=1,
            line_dash="dash",
            opacity=0.55,
        )

    episode_no = int(episode_df["episode_no"].iloc[0]) if "episode_no" in episode_df else 0

    fig.update_layout(
        title=f"Rolling reward rate — Episode {episode_no + 1}",
        xaxis_title="Step",
        yaxis_title="Step reward total",
        height=360,
        margin={"l": 40, "r": 20, "t": 55, "b": 40},
        legend={"orientation": "h", "y": -0.25},
    )

    return fig


def add_policy_background_vrects(
    fig: go.Figure,
    episode_df: pd.DataFrame,
    *,
    policy_column: str,
) -> None:
    if policy_column not in episode_df or "step" not in episode_df:
        return

    policy_df = episode_df.sort_values("step").copy()

    if policy_df.empty:
        return

    steps = policy_df["step"].to_numpy(dtype=float)
    policies = policy_df[policy_column].astype(str).to_numpy()
    unique_policies = sorted(policy_df[policy_column].astype(str).unique())

    palette = qualitative.Plotly
    policy_colours = {
        policy: palette[index % len(palette)]
        for index, policy in enumerate(unique_policies)
    }

    step_delta = float(np.median(np.diff(steps))) if len(steps) > 1 else 1.0
    segment_start_index = 0

    for index in range(1, len(policy_df) + 1):
        reached_end = index == len(policy_df)
        policy_changed = not reached_end and policies[index] != policies[segment_start_index]

        if reached_end or policy_changed:
            policy = policies[segment_start_index]
            start_step = steps[segment_start_index] - 0.5 * step_delta
            end_step = steps[index - 1] + 0.5 * step_delta

            fig.add_vrect(
                x0=start_step,
                x1=end_step,
                fillcolor=policy_colours[policy],
                opacity=0.08,
                line_width=0,
                layer="below",
                annotation_text=policy,
                annotation_position="top left",
            )

            segment_start_index = index


def make_query_trajectory_figure(
    database: dict[str, Any],
    steps_df: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()

    umap_embeddings = database["umap_embeddings"]

    fig.add_trace(
        go.Scattergl(
            x=umap_embeddings[:, 0],
            y=umap_embeddings[:, 1],
            mode="markers",
            marker={"size": 4, "color": "lightgrey", "opacity": 0.18},
            name="historical database",
        )
    )

    query_df = get_query_df(steps_df)

    if not query_df.empty:
        fig.add_trace(
            go.Scatter(
                x=query_df["query_umap_x"],
                y=query_df["query_umap_y"],
                mode="lines",
                line={"width": 2},
                name="live query path",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=query_df["query_umap_x"],
                y=query_df["query_umap_y"],
                mode="markers",
                marker={
                    "size": 8,
                    "color": query_df["step"],
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "Step"},
                    "line": {"width": 1, "color": "black"},
                },
                name="live queries",
            )
        )

        switch_df = query_df[get_switch_mask(query_df)]

        if not switch_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=switch_df["query_umap_x"],
                    y=switch_df["query_umap_y"],
                    mode="markers",
                    marker={
                        "size": 14,
                        "color": "red",
                        "symbol": "x",
                        "line": {"width": 2, "color": "black"},
                    },
                    name="committed switch",
                )
            )

    episode_no = int(steps_df["episode_no"].iloc[-1]) if not steps_df.empty else 0

    fig.update_layout(
        title=f"Live query UMAP trajectory — Episode {episode_no + 1}",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        height=620,
        margin={"l": 40, "r": 20, "t": 55, "b": 40},
    )

    return fig


def make_database_and_queries_figure(
    database: dict[str, Any],
    steps_df: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()

    umap_embeddings = database["umap_embeddings"]
    database_policy_keys = np.asarray(database["policy_keys"]).astype(str)

    policy_markers = {
        "flat": "circle",
        "ramp": "triangle-up",
        "uneven": "square",
    }

    policy_labels = sorted(np.unique(database_policy_keys).astype(str))

    for policy in policy_labels:
        mask = database_policy_keys == policy

        fig.add_trace(
            go.Scattergl(
                x=umap_embeddings[mask, 0],
                y=umap_embeddings[mask, 1],
                mode="markers",
                marker={
                    "size": 5,
                    "opacity": 0.18,
                    "symbol": policy_markers.get(policy, "circle"),
                },
                name=f"database {policy}",
            )
        )

    query_df = get_query_df(steps_df)

    if not query_df.empty and "current_policy" in query_df:
        for policy in sorted(query_df["current_policy"].astype(str).unique()):
            mask = query_df["current_policy"].astype(str) == policy

            fig.add_trace(
                go.Scatter(
                    x=query_df.loc[mask, "query_umap_x"],
                    y=query_df.loc[mask, "query_umap_y"],
                    mode="markers",
                    marker={
                        "size": 10,
                        "symbol": policy_markers.get(policy, "circle"),
                        "line": {"width": 1, "color": "black"},
                    },
                    name=f"query {policy}",
                )
            )

    episode_no = int(steps_df["episode_no"].iloc[-1]) if not steps_df.empty else 0

    fig.update_layout(
        title=f"Comparator UMAP space — Episode {episode_no + 1}",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        height=620,
        margin={"l": 40, "r": 20, "t": 55, "b": 40},
    )

    return fig


def make_vote_count_figure(latest_step: dict[str, Any] | None) -> go.Figure:
    fig = go.Figure()

    if not latest_step:
        fig.update_layout(title="Policy vote counts", height=320)
        return fig

    vote_counts = latest_step.get("policy_vote_counts") or {}

    if not isinstance(vote_counts, dict) or len(vote_counts) == 0:
        fig.update_layout(title="Policy vote counts", height=320)
        return fig

    policies = list(vote_counts.keys())
    counts = [vote_counts[policy] for policy in policies]

    fig.add_trace(
        go.Bar(
            x=counts,
            y=policies,
            orientation="h",
            name="votes",
        )
    )

    fig.update_layout(
        title="Latest comparator policy vote counts",
        xaxis_title="Candidate count",
        yaxis_title="Policy",
        height=320,
        margin={"l": 60, "r": 20, "t": 55, "b": 40},
    )

    return fig


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    st.set_page_config(
        page_title="Comparator Dashboard",
        layout="wide",
    )

    st.title("Live Comparator Dashboard")

    mqtt_topic = f"quaid/comparator/r{args.mqtt_queue}/telemetry"

    with st.sidebar:
        st.header("Inputs")
        st.write(f"**Comparator path**: `{args.comparator_path}`")
        st.write(f"**MQTT**: `{args.mqtt_host}:{args.mqtt_port}`")
        st.write(f"**Topic**: `{mqtt_topic}`")
        st.write(f"**Rolling window**: `{args.rolling_window}`")

    try:
        database = load_comparator_database(args.comparator_path)
    except Exception as exc:
        st.error(str(exc))
        return

    try:
        live_state = get_live_state(
            mqtt_host=args.mqtt_host,
            mqtt_port=args.mqtt_port,
            mqtt_topic=mqtt_topic,
            max_steps=args.max_steps,
        )
    except Exception as exc:
        st.error(f"Could not connect to MQTT broker: {exc}")
        return

    snapshot = live_state.snapshot()
    steps_df = pd.DataFrame(snapshot.steps)

    if not steps_df.empty:
        numeric_columns = [
            "episode_no",
            "step",
            "step_reward_total",
            "episode_reward_total",
            "query_umap_x",
            "query_umap_y",
            "query_local_reward_mean",
            "candidate_count",
            "selected_policy_fraction",
            "vote_margin",
            "switch_committed",
        ]

        for column in numeric_columns:
            if column in steps_df:
                steps_df[column] = pd.to_numeric(steps_df[column], errors="coerce")

    latest_step = snapshot.last_message if snapshot.last_message else None
    latest_step_row = snapshot.steps[-1] if len(snapshot.steps) > 0 else latest_step

    status_cols = st.columns(6)

    status_cols[0].metric("MQTT", "connected" if snapshot.connected else "disconnected")
    status_cols[1].metric("Episode", "—" if snapshot.current_episode_no is None else snapshot.current_episode_no + 1)
    status_cols[2].metric("Step", "—" if latest_step_row is None else latest_step_row.get("step", "—"))
    status_cols[3].metric("Current policy", "—" if latest_step_row is None else latest_step_row.get("current_policy", "—"))
    status_cols[4].metric("Next policy", "—" if latest_step_row is None else latest_step_row.get("next_policy", "—"))
    status_cols[5].metric(
        "Episode reward",
        "—" if latest_step_row is None else f"{float(latest_step_row.get('episode_reward_total', 0.0)):.3f}",
    )

    if snapshot.last_received_time is None:
        st.info("Waiting for comparator telemetry...")
    else:
        age = time.time() - snapshot.last_received_time
        st.caption(f"Last telemetry message received {age:.2f} seconds ago.")

    left_col, right_col = st.columns([1.15, 1.0])

    with left_col:
        st.plotly_chart(
            make_query_trajectory_figure(database, steps_df),
            width="stretch",
        )

    with right_col:
        st.plotly_chart(
            make_reward_rate_figure(steps_df, args.rolling_window),
            width="stretch",
        )

        st.plotly_chart(
            make_vote_count_figure(latest_step_row),
            width="stretch",
        )

    st.plotly_chart(
        make_database_and_queries_figure(database, steps_df),
        width="stretch",
    )

    with st.expander("Switch log", expanded=True):
        switch_df = steps_df[steps_df.get("switch_committed", 0).astype(int) == 1] if not steps_df.empty else pd.DataFrame()

        if switch_df.empty:
            st.write("No committed switches in the current episode yet.")
        else:
            columns = [
                column
                for column in [
                    "step",
                    "current_policy",
                    "next_policy",
                    "active_policy_after_step",
                    "selected_policy_fraction",
                    "vote_margin",
                    "candidate_count",
                ]
                if column in switch_df
            ]
            st.dataframe(switch_df[columns].sort_values("step"), width="stretch")

    with st.expander("Latest telemetry payload"):
        st.json(latest_step_row or {})

    if args.refresh_ms > 0:
        time.sleep(args.refresh_ms / 1000.0)
        st.rerun()


if __name__ == "__main__":
    main()

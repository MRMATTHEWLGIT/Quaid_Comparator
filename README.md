# Quaid_Comparator

## 1.0 Building Comparator Assets

The comparator system requires a precomputed database of historical rollout embeddings.

The `build_comparator.py` script:

- extracts embeddings from a trained GRU model,
- exports the GRU encoder to ONNX,
- fits either a standard or parametric UMAP model,
- generates the comparator database used during runtime.

---

### 1.1 Testset Location

Historical rollout testsets should be placed inside:

```text
testsets/
```

Example:

```text
testsets/
    test_O+A_O_DbP26_SeqL100_Str10_20260504_161211.npz
```

---

### 1.2 Example Usage

#### 1.2.1 Standard UMAP

```bash
python build_comparator.py \
    --gru-path Models/Base/trained_model_GRU.pth \
    --testset-path testsets/test_dataset.npz \
    --umap-kind standard
```

#### 1.2.2 Parametric UMAP

```bash
python build_comparator.py \
    --gru-path Models/Base/trained_model_GRU.pth \
    --testset-path testsets/test_dataset.npz \
    --umap-kind parametric \
    --parametric-epochs 20 \
    --parametric-batch-size 4096
```

---

### 1.3 Output Directory

Generated comparator assets are saved into:

```text
models/comparator/
```

Each run creates a timestamped folder:

```text
models/comparator/comparator_standard_2026-05-12T17-24-30/
```

or:

```text
models/comparator/comparator_parametric_2026-05-12T17-24-30/
```

---

### 1.4 Generated Assets

#### 1.4.1 Standard UMAP

```text
comparator_database.npz
embedding_gru_encoder.onnx
embedding_gru_stats.npz
umap_model.pkl
metadata.json
comparator_assets.yaml
```

#### 1.4.2 Parametric UMAP

```text
comparator_database.npz
embedding_gru_encoder.onnx
embedding_gru_stats.npz
parametric_umap_encoder.onnx
metadata.json
comparator_assets.yaml
```

---

### 1.5 Command Line Arguments

| Argument | Description | Default |
|---|---|---|
| `--gru-path` | Path to trained embedding GRU checkpoint (`.pth`) | Required |
| `--testset-path` | Path to rollout testset (`.npz`) | Required |
| `--umap-kind` | UMAP type: `standard` or `parametric` | Required |
| `--output-root` | Root output directory | `models` |
| `--n-neighbors` | UMAP nearest neighbours | `15` |
| `--min-dist` | UMAP minimum embedding distance | `0.1` |
| `--metric` | UMAP distance metric | `euclidean` |
| `--random-state` | Random seed for reproducibility | `42` |
| `--parametric-epochs` | Parametric UMAP training epochs | `10` |
| `--parametric-batch-size` | Parametric UMAP batch size | `4096` |

---

### 1.6 Runtime Usage

The generated assets can then be referenced from `comparator.yaml`
during comparator inference.


## 2.0 Live Dashboard

The comparator dashboard provides a live Streamlit view of the current comparator episode using MQTT telemetry. It overlays the live query embeddings onto the saved comparator UMAP database and displays episode-level reward and switching behaviour.

Before launching the dashboard, make sure the MQTT broker is running and that the comparator assets have already been built with `build_comparator.py`.

### 2.1 Launch Dashboard

```bash
python -m dashboard \
    --comparator-path models\comparator\comparator_parametric_2026-05-17T22-30-15\ \
    --mqtt-host localhost \
    --mqtt-port 1883 \
    --mqtt-queue 100 \
    --refresh-ms 1000
```

The `--comparator-path` argument should point to the comparator asset folder containing:

```text
comparator_database.npz
metadata.json
embedding_gru_encoder.onnx
parametric_umap_encoder.onnx
```

The `--mqtt-queue` argument is used to subscribe to the corresponding comparator telemetry topic:

```text
quaid/comparator/r100/telemetry
```

### 2.2 Run Comparator with Dashboard Telemetry

In a separate terminal, run the comparator with the same MQTT queue:

```bash
python main.py \
    --env-config config/quaid_sim.yaml \
    --comparator-config config/comparator.yaml \
    --mqtt-queue 100 \
    --dashboard-mqtt
```

The dashboard can be launched before the comparator program. It will wait for MQTT telemetry and reset its plots at the start of each new episode.

### 2.3 Useful Options

| Argument | Description |
|---|---|
| `--comparator-path` | Path to the comparator asset folder |
| `--mqtt-host` | MQTT broker host, usually `localhost` |
| `--mqtt-port` | MQTT broker port, usually `1883` |
| `--mqtt-queue` | Queue number used to form `quaid/comparator/r<queue>/telemetry` |
| `--refresh-ms` | Dashboard redraw interval in milliseconds |

## 3.0 Running the Comparator

The main comparator runtime is launched through `main.py`. It loads the Quaid environment configuration, loads the comparator policy configuration, runs the episode playbook, and optionally publishes live dashboard telemetry over MQTT.

### 3.1 Basic Command

```bash
python main.py \
    --env-config config/quaid_sim.yaml \
    --comparator-config config/comparator.yaml \
    --mqtt-queue 100 \
    --max-steps 500
```

On Windows PowerShell, the same command can be written as:

```powershell
python main.py `
    --env-config config/quaid_sim.yaml `
    --comparator-config config/comparator.yaml `
    --mqtt-queue 100 `
    --max-steps 500
```

The `--env-config` argument is required and should point to the Quaid environment YAML file, such as `config/quaid_sim.yaml`. This file defines the simulation/MQTT environment settings, including the MQTT broker address, queue number, robot settings, observation configuration, and reward configuration.

The `--comparator-config` argument points to the comparator YAML file, such as `config/comparator.yaml`. This file defines which policies are available, where the comparator assets are located, and how the comparator selects policies.

### 3.2 Running with Dashboard Telemetry

To publish live MQTT telemetry for the Streamlit dashboard, add `--dashboard-mqtt`:

```bash
python main.py \
    --env-config config/quaid_sim.yaml \
    --comparator-config config/comparator.yaml \
    --mqtt-queue 100 \
    --max-steps 500 \
    --dashboard-mqtt
```

By default, this publishes dashboard telemetry to:

```text
quaid/comparator/r100/telemetry
```

where `100` comes from `--mqtt-queue 100`.

A custom dashboard topic can also be provided:

```bash
python main.py \
    --env-config config/quaid_sim.yaml \
    --comparator-config config/comparator.yaml \
    --mqtt-queue 100 \
    --dashboard-mqtt \
    --dashboard-mqtt-topic quaid/comparator/r100/telemetry
```

### 3.3 Comparator Configuration

The comparator is configured using `config/comparator.yaml`.

A typical comparator configuration contains:

```yaml
initial_policy: "ramp"

episode_playbook:
  - comparator
  - comparator
  - comparator

policies:
  flat:
    model_path: "models/onnx/aug_act_net_QuaidSIM-Flat_RA-TD3_+305.334_475000.onnx"

  ramp:
    model_path: "models/onnx/aug_act_net_QuaidSIM-Ramp_RA-TD3_+301.146_475000.onnx"

  uneven:
    model_path: "models/onnx/aug_act_net_QuaidSIM-Uneven_RA-TD3_+348.233_400099.onnx"

comparator_assets_dir: "models/comparator/sim_dynamic_comparator_parametric_2026-05-17T22-30-15"

hyperparameters:
  sequence_length: 99
  k_signature: 80
  k_reward: 40
  min_reward_gain_percent: 5.0
  min_distribution_overlap: 0.75
  min_pairwise_agreement: 1.00
  include_current_policy_candidates: true
  min_vote_candidates: 10
  min_vote_fraction: 0.50
  min_vote_margin: 0.10
  required_consecutive_policy_votes: 3
```

#### 3.3.1 Key Fields

| Field | Description |
|---|---|
| `initial_policy` | Policy used at the start of each comparator episode before any switch is made |
| `episode_playbook` | Determines which policy mode to run for each episode |
| `policies` | Lists the available policies and their ONNX model paths |
| `comparator_assets_dir` | Path to the comparator asset folder generated by `build_comparator.py` |
| `hyperparameters` | Controls nearest-neighbour matching, candidate filtering, and switching behaviour |

The `episode_playbook` controls the run structure. For example:

```yaml
episode_playbook:
  - flat
  - ramp
  - uneven
  - comparator
  - comparator
  - comparator
```

This runs three fixed-policy episodes followed by three comparator-controlled episodes. If an entry is a policy name, that episode uses only that policy. If an entry is `comparator`, the comparator is allowed to select and switch between the configured policies.

### 3.4 Environment Configuration

The environment configuration is provided with `--env-config`, for example:

```bash
--env-config config/quaid_sim.yaml
```

This file defines the Quaid environment and MQTT settings. Important fields include:

```yaml
ports:
  mqtt_server_ip: "tcp://127.0.0.1:1883"
  mqtt_queue_no: "100"

robot:
  sim: 1
  env_logger: 1
  max_steps: 500
  step_time: 50
```

The `--mqtt-queue` command-line argument can override the `mqtt_queue_no` value in the environment YAML:

```bash
--mqtt-queue 100
```

This is useful when running multiple Quaid instances or matching the comparator runtime with the Streamlit dashboard.

### 3.5 Useful Runtime Arguments

| Argument | Description |
|---|---|
| `--env-config`, `-c` | Required Quaid environment YAML file |
| `--comparator-config` | Comparator YAML file |
| `--initial-policy` | Optional override for the initial policy |
| `--mqtt-queue`, `-q` | Override the MQTT queue number from the environment YAML |
| `--gru-path`, `-g` | Optional global GRU override path |
| `--max-steps`, `-s` | Maximum number of steps per episode |
| `--step-delay-ms` | Optional extra delay between environment steps |
| `--output-root` | Parent directory for timestamped run outputs |
| `--no-logger` | Disable per-step SQLite logging |
| `--verbose`, `-v` | Enable more detailed logging |
| `--dashboard-mqtt` | Publish live telemetry for the Streamlit dashboard |
| `--dashboard-mqtt-topic` | Optional custom dashboard MQTT telemetry topic |

## 4.0 Comparator Run Analysis

The comparator analysis script is located at `analysis/analyse_comparator_run.py`. It is used after a comparator run has finished to load the saved runtime logs, enrich the comparator step data, and generate summary plots for each episode.

### 4.1 Basic Command

```bash
python analysis/analyse_comparator_run.py \
    --data-path data\2026-05-18T20-18-27_FFFCCC
```

The `--data-path` argument should point to a saved comparator run directory produced by `main.py`.

By default, the analysis outputs are saved to:

```text
<data-path>/analysis_results
```

For example:

```text
data/2026-05-18T20-18-27_FFFCCC/analysis_results
```

### 4.2 Custom Output Directory

A custom output directory can be provided with `--output-dir`:

```bash
python analysis/analyse_comparator_run.py \
    --data-path data\2026-05-18T20-18-27_FFFCCC \
    --output-dir analysis_outputs\run_001
```

If `--output-dir` is not provided, the script automatically creates an `analysis_results` folder inside the selected run directory.

### 4.3 Required Run Directory Files

The selected run directory should contain the saved comparator runtime files created by `main.py`, including:

```text
comparator.yaml
env.yaml
inference_times.db
Quaid_*.sqlite
```

The script uses `comparator.yaml` to locate the comparator asset folder through:

```yaml
comparator_assets_dir: "models/comparator/comparator_parametric_2026-05-17T22-30-15"
```

That comparator asset folder should contain:

```text
comparator_database.npz
metadata.json
```

These files are used to load the historical UMAP database and overlay the live query embeddings from the run.

### 4.4 Generated Outputs

The script saves an enriched comparator step CSV:

```text
analysis_results/comparator_steps_enriched.csv
```

This CSV contains the original logged comparator step data plus additional parsed and derived fields, such as policy vote counts, candidate filter counts, and candidate UMAP summary values.

The script also creates aggregate plots in:

```text
analysis_results/aggregate
```

including:

```text
final_reward_per_episode.png
cumulative_reward_curves.png
policy_occupancy_per_episode.png
aggregate_reward_rate.png
```

### 4.5 Per-Episode Plots

For each episode, the script creates a separate folder:

```text
analysis_results/episodes/episode_000
analysis_results/episodes/episode_001
analysis_results/episodes/episode_002
```

Each episode folder contains plots such as:

```text
umap_database_with_queries.png
query_umap_trajectory.png
policy_vote_counts.png
policy_timeline.png
switch_reward_delta.png
reward_rate.png
reward_components.png
```

These plots are useful for inspecting how the comparator moved through the UMAP embedding space, when it switched policies, and how the reward components changed throughout the episode.

### 4.6 Useful Arguments

| Argument | Description |
|---|---|
| `--data-path` | Path to the saved comparator run directory |
| `--output-dir` | Optional custom folder for analysis outputs |

### 4.7 Typical Workflow

A typical workflow is:

```text
1. Run the comparator with main.py
2. Locate the saved run directory inside data/
3. Run analysis/analyse_comparator_run.py with --data-path
4. Inspect analysis_results/comparator_steps_enriched.csv
5. Review aggregate and per-episode plots
```

Example:

```bash
python main.py \
    --env-config config/quaid_sim.yaml \
    --comparator-config config/comparator.yaml \
    --mqtt-queue 100 \
    --max-steps 500 \
    --dashboard-mqtt

python analysis/analyse_comparator_run.py \
    --data-path data\2026-05-18T20-18-27_FFFCCC
```



## 5.0 Third-Party Code and Acknowledgements

Portions of this project are derived from or inspired by the following
open-source repository:

- [esp-dl-quant-icra2026](https://github.com/real-world-drl/esp-dl-quant-icra2026)  
  Licensed under the MIT License.

A modified vendored copy of the Quaid environment used by this project is
located in:

```text
third_party/quaid_env/
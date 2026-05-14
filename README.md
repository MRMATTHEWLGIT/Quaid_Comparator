# Quaid_Comparator

# Building Comparator Assets

The comparator system requires a precomputed database of historical rollout embeddings.

The `build_comparator.py` script:

- extracts embeddings from a trained GRU model,
- exports the GRU encoder to ONNX,
- fits either a standard or parametric UMAP model,
- generates the comparator database used during runtime.

---

## Testset Location

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

## Example Usage

### Standard UMAP

```bash
python build_comparator.py \
    --gru-path Models/Base/trained_model_GRU.pth \
    --testset-path testsets/test_dataset.npz \
    --umap-kind standard
```

### Parametric UMAP

```bash
python build_comparator.py \
    --gru-path Models/Base/trained_model_GRU.pth \
    --testset-path testsets/test_dataset.npz \
    --umap-kind parametric \
    --parametric-epochs 20 \
    --parametric-batch-size 4096
```

---

## Output Directory

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

## Generated Assets

### Standard UMAP

```text
comparator_database.npz
embedding_gru_encoder.onnx
embedding_gru_stats.npz
umap_model.pkl
metadata.json
comparator_assets.yaml
```

### Parametric UMAP

```text
comparator_database.npz
embedding_gru_encoder.onnx
embedding_gru_stats.npz
parametric_umap_encoder.onnx
metadata.json
comparator_assets.yaml
```

---

## Command Line Arguments

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

## Runtime Usage

The generated assets can then be referenced from `comparator.yaml`
during comparator inference.

## Third-Party Code and Acknowledgements

Portions of this project are derived from or inspired by the following
open-source repository:

- [esp-dl-quant-icra2026](https://github.com/real-world-drl/esp-dl-quant-icra2026)  
  Licensed under the MIT License.

A modified vendored copy of the Quaid environment used by this project is
located in:

```text
third_party/quaid_env/
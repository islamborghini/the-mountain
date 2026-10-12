# The Mountain

A standalone ensemble for predicting protein insertion/deletion effects. It runs a fixed scoring rule without an LLM, research loop, API key, or experimental fitness labels.

Two explicit inference profiles are available:

| Profile | Inputs | What runs locally |
|---|---|---|
| `sequence_6` | Mutated sequences grouped by protein | Four RITA models, ProtGPT2, ESM2-650M, then the ensemble |
| `original_24` | Scores from 23 published predictors plus ESM2 | The original full ensemble; component scores may be supplied or downloaded for the frozen benchmark |

`sequence_6` is a smaller model with its own measured performance. The original 24-signal method includes family/MSA/retrieval predictors and cannot generate every component from sequences alone.

## Install

Use Python 3.10–3.13 (tested with Python 3.12). Create a virtual environment, then install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[inference,verification,test]'
```

For supplied-score inference only, `python -m pip install .` is sufficient. CUDA, Apple MPS, and CPU are supported for sequence inference. The six default checkpoints are large and load sequentially; first use downloads the weights into the Hugging Face cache. All weight revisions are pinned. RITA also executes a reviewed custom architecture from its pinned checkpoint revision.

## Predict new sequences

Input CSV requires `protein_id`, `variant_id`, and `mutated_sequence`. IDs must be unique within each protein. Sequences must contain uppercase canonical amino acids. See `examples/sequences.csv`.

```bash
mountain-predict --sequences examples/sequences.csv --output predictions.csv
```

This runs all six models. `--device cpu`, `--device cuda`, or `--device mps` selects a device. `--cache-dir` sets the resumable feature cache. For a smaller smoke run:

```bash
mountain-predict --sequences examples/sequences.csv --models RITA_s,ESM2_650M --device cpu --output smoke.csv
```

A subset is identified separately in the manifest and does not inherit either full profile's benchmark result. ESM2 rejects sequences longer than 1022 residues by default; `--length-policy truncate` explicitly scores their first 1022 residues. Causal models score nonoverlapping 1023-residue chunks in both directions. A chunk with fewer than two tokenizer tokens is rejected because next-token scoring is undefined.

Output contains the original identifiers and sequences plus `mountain_score`. Higher scores predict greater fitness. Scores are relative ranks within each protein's supplied variant group: changing that group can change scores. They are not calibrated probabilities or scores comparable across proteins. Average tied ranks make identical inputs receive identical votes, independent of row order.

`predictions.csv.manifest.json` records input/output hashes, profile, checkpoints, runtime versions, length policy, and protein groups. Feature caches bind exact ordered sequences, checkpoint revisions, scoring policy, precision, device, and runtime versions.

## Predict from existing component scores

```bash
mountain-predict --signals examples/signals.csv --output predictions.csv
```

Use the exact component names in `model.SIGNALS`; higher component values must indicate greater fitness. Include `protein_id` and `variant_id`; sequence is optional. All-NaN columns mean unavailable components. Partially missing or infinite scores are rejected. Labels and PoET are rejected as prediction inputs. Supplied scores do not imply reproducible component generation; their source remains the user's responsibility.

The Python API is `model.score_variants(matrix, signal_names)`, called separately for each protein. Importing model or CLI modules performs no downloads or filesystem writes.

## Reproduce the frozen benchmark

The committed snapshot fixes ProteinGym v1.3 DMS-Indels to 66 assays. The archive, reference, each input CSV, and ordered sequences are hash-checked. A historical CAPSD assay uses the causal/published consensus without ESM2; all of its variants still receive predictions.

Original ensemble with fresh ESM2 and the published component scores:

```bash
mountain-benchmark --root proteingym_artifact --profile original_24 --generate-esm2
mountain-verify --root proteingym_artifact
```

To recompute all six native components directly from benchmark sequences:

```bash
mountain-benchmark --root native_artifact --profile sequence_6 --generate-components --device cuda
mountain-verify --root native_artifact
```

Native reproduction is expensive and resumable. It uses float32 ESM2; historical arrays used float16, so fresh metrics may differ. Its benchmark length policy truncates ESM2 prefixes as the original did; prediction defaults to an explicit error.

Existing ESM2 vectors require JSON sidecars binding their ordered sequences and array hashes. For the historical `cache_combo.zip`, `--legacy-cache-zip /path/to/cache_combo.zip` explicitly migrates the frozen archive, verifying stored labels and component alignment. Migration records that its historical ESM2 sequence order/checkpoint provenance cannot independently be established. A cache from the source repository can be used directly; neither its notebooks nor research services run. Never describe migration as fresh component inference.

Use separate `--root` directories for profiles. `python main.py` remains a benchmark entry point; pass the same options as `mountain-benchmark`.

Each run produces:

- `results/predictions/`: label-free variant scores for every assay.
- `results/labels/`: labels kept separately for evaluation.
- `results/official_merged/`: full input tables for ProteinGym's evaluator.
- `results/assay_scores.csv`, `summary.csv`, and seeded protein-cluster bootstrap results.
- `results/run_manifest.json`: input/output hashes, feature provenance, runtime versions, and settings.

`mountain-verify` downloads and hash-checks the unmodified official evaluator at commit `144fe22b07dfaeec2b366f2346203a9838a55b4c`. It checks all 66 assays were evaluated and compares per-assay and aggregate Spearman scores with local results. The official evaluator also reports its other standard metrics. It does not submit a leaderboard entry.

For a single prediction/label CSV pair with matching `variant_id` and `mutated_sequence` columns:

```bash
mountain-evaluate --predictions predictions.csv --labels labels.csv --column mountain_score
```

## Verified historical-input results

| Profile | Unrounded aggregate Spearman | Official reported Spearman | Protein-cluster 95% CI for improvement over PoET |
|---|---:|---:|---:|
| Original 24 | 0.557896 | 0.558 | +0.0196 to +0.0754 |
| Sequence 6 | 0.541160 | 0.541 | -0.0063 to +0.0579 |
| PoET | 0.516897 | 0.517 | reference |

Both profiles were evaluated on all 66 assays using published causal scores and historical ESM2 arrays. The unmodified official evaluator confirmed per-assay agreement and the reported rounded aggregates. ProteinGym rounds assay metrics to three decimals before aggregation; the local report includes both rounded and unrounded values.

The original README's 0.5604 is superseded by the reproduced 0.557896 result. The six-model point estimate exceeds PoET; its uncertainty interval includes zero, so statistically established superiority is not claimed. The historical protein-disjoint split was used during development and is not untouched independent validation. Removing tuned constants does not erase label-informed method selection.

Evidence is under `validation/original_24/` and `validation/sequence_6/`. CPU smoke inference using real RITA-small and ESM2-650M checkpoints succeeded. RITA-small matched three published scores within 4.8e-7. A fresh six-checkpoint, full-benchmark run and official leaderboard acceptance remain unverified.

## Method and checks

Rank each available component within its protein's variant group, take the median of the published/causal group, combine that vote equally with ESM2 using a geometric mean, and apply `1 / (1 + variance_of_component_ranks)` as an agreement factor. Missing ESM2 falls back to the other group's consensus. PoET is a reference only.

```bash
python -m pytest -q
```

Tests cover tied ranks and row permutations, malformed input and missing scores, group isolation, labels excluded from inference, atomic output, native scoring conventions, cache invalidation/resumption, archive hashes/alignment, and benchmark export. Primary scoring references: [RITA](https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/proteingym/baselines/rita/compute_fitness.py), [ProtGPT2](https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/proteingym/baselines/protgpt2/compute_fitness.py), and [ProteinGym submission requirements](https://github.com/OATML-Markslab/ProteinGym#new-baselines).

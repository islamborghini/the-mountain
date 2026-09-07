# The Mountain

A leaderboard-beating ensemble for the [ProteinGym DMS-Indels benchmark](https://proteingym.org/benchmarks).

The method reaches an average Spearman correlation of **0.5604** across the 66-assay benchmark. The current public leader, PoET, is listed at **0.517**; this pipeline reproduces it at **0.5169** using the same hierarchical aggregation.


## Results

| Method | Full benchmark | Development | Protein-disjoint held-out |
|---|---:|---:|---:|
| PoET | 0.5169 | 0.5376 | 0.4767 |
| Equal-weight ensemble | 0.5248 | 0.5439 | 0.4995 |
| **The Mountain** | **0.5604** | **0.5720** | **0.5658** |
| Without agreement weighting | 0.5597 | 0.5715 | 0.5632 |
| Without signal-family separation | 0.5170 | 0.5328 | 0.5048 |

The full-benchmark improvement over PoET is **+0.0435**. A 5,000-sample protein-cluster bootstrap gives a 95% interval of **+0.0236 to +0.0763**, with the candidate ahead of PoET in every replicate.

## Method

The pipeline combines 23 published non-PoET scores with an ESM2-650M feature:

1. Convert every signal to within-assay ranks.
2. Collapse the correlated published models into one median consensus.
3. Give that consensus and ESM2 equal structural weight using a geometric mean.
4. Apply a small penalty when the input models disagree.
5. Evaluate per-assay Spearman correlations using ProteinGym's protein- and selection-aware aggregation.

The ablation is the key result: removing the separation between the published-model consensus and ESM2 drops the full score from **0.5604** to **0.5170**. The gain comes primarily from balancing distinct sources of evidence, not from the disagreement penalty.

PoET is used only as an external reference and is never an ensemble input.

## Run

Python 3.10+ and an NVIDIA CUDA GPU are required for the first run.

```bash
python -m pip install numpy pandas requests scipy torch transformers tqdm
python main.py
```

The script downloads and validates the ProteinGym v1.3 inputs, generates missing ESM2 arrays, builds aligned caches, runs the ablations and bootstrap, and writes:

```text
proteingym_artifact/results/assay_scores.csv
proteingym_artifact/results/summary.csv
proteingym_artifact/results/bootstrap_clean_full_vs_PoET.csv
proteingym_artifact/results/manifest.json
```

The expensive ESM2 pass is resumable. Existing arrays under `proteingym_artifact/data/esm2_650m/` are reused automatically.

## Evaluation details

- The 66 assays are split by protein identity into 54 development and 12 held-out assays.
- ESM2 is available for 65 assays. The oversized CAPSD assay uses the published-model consensus fallback.
- The ESM2 feature is mean observed-token confidence from one unmasked forward pass. It is not masked pseudo-log-likelihood.
- Bootstrap samples are clustered by protein so assays from the same protein remain dependent.
- Seeds, source hashes, feature names, package versions, and scoring settings are recorded in `manifest.json`.

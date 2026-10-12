# Frozen benchmark inputs

`DMS_indels.csv` is the 66-assay ProteinGym v1.3 indel reference snapshot.
`benchmark_manifest.json` pins its SHA-256, the published score archive, every
assay CSV and its ordered sequence list. The benchmark rejects changed assay
membership or content rather than silently following ProteinGym's `main` branch.

Published scores: https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/zero_shot_indels_scores.zip

Official evaluator: https://github.com/OATML-Markslab/ProteinGym/tree/144fe22b07dfaeec2b366f2346203a9838a55b4c

Ordered sequence hashes use SHA-256 of UTF-8 encoded
`json.dumps(list(sequences), ensure_ascii=True, separators=(',', ':'))`.

The optional original cache archive is pinned separately. Explicit migration
checks every experimental label within 1e-12 decimal parsing tolerance and each
published component within two float32 rounding units before binding its ESM2
array to these sequences. The old cache did not contain ESM2 sequence/checkpoint
metadata: these checks cannot independently prove historical ESM2 provenance.
Every migrated array records this limitation in its sidecar and run manifest.
Newly generated feature arrays carry checkpoint and scoring metadata.

The historical protein split is reproduced for audit only. The method was
selected using this benchmark; neither split is claimed as untouched validation.
The all-assay comparison is a benchmark reproduction, not an independent test of
generalization. The six-model inference profile has its own evaluation and must
not inherit the full ensemble's reported result.

ProteinGym rounds each assay metric to 3 decimal places before hierarchical
aggregation. `summary.csv` exposes both unrounded and official-rounded results.
The paired protein bootstrap in this repository is a separate analysis; it is
not ProteinGym's bootstrap standard-error calculation.

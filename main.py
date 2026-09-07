# ProteinGym DMS-Indels ensemble and evaluation pipeline.

from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import platform
import random

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

# Output layout; change ROOT to reuse a persistent cache.
ROOT = Path.cwd() / "proteingym_artifact"
DATA_DIR = ROOT / "data"
SCORE_DIR = DATA_DIR / "published_scores"
ESM2_DIR = DATA_DIR / "esm2_650m"
ESM2_SEARCH_DIRS = (
    ESM2_DIR,
    Path("/content/pg_new_signals/ESM2_650M"),
    Path("/content/drive/MyDrive/proteingym_indels/pg_new_signals/ESM2_650M"),
)
CACHE_DIR = ROOT / "cache"
RESULT_DIR = ROOT / "results"

for path in (DATA_DIR, SCORE_DIR, ESM2_DIR, CACHE_DIR, RESULT_DIR):
    path.mkdir(parents=True, exist_ok=True)

REFERENCE_URL = (
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
    "reference_files/DMS_indels.csv"
)
SCORES_URL = (
    "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/"
    "zero_shot_indels_scores.zip"
)

REFERENCE_PATH = DATA_DIR / "DMS_indels.csv"
SCORES_ZIP = DATA_DIR / "zero_shot_indels_scores.zip"

# Expensive and stochastic settings live here for reproducibility.
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
GENERATE_ESM2 = True
ESM2_MAX_RESIDUES = 1022
ESM2_BATCH_TOKENS = 1500
SPLIT_SEED = 0
BOOTSTRAP_SEED = 0
N_BOOTSTRAPS = 5000

# PoET is deliberately excluded from the ensemble inputs.
PUBLISHED_SIGNALS = [
    "Unirep", "Wavenet",
    "RITA_s", "RITA_m", "RITA_l", "RITA_xl",
    "Progen2_small", "Progen2_medium", "Progen2_base",
    "Progen2_large", "Progen2_xlarge",
    "ProtGPT2",
    "Tranception_S_no_retrieval",
    "Tranception_M_no_retrieval",
    "Tranception_L_no_retrieval",
    "Tranception_S", "Tranception_M", "Tranception_L",
    "TranceptEVE_S", "TranceptEVE_M", "TranceptEVE_L",
    "HMM", "Provean",
]
SIGNALS = PUBLISHED_SIGNALS + ["ESM2_650M"]

SKIP_ESM2 = {"CAPSD_AAV2S_Sinai_2021_designed_indels"}

def download(url, destination, chunk_size=1 << 20):
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size:
        return destination

    # Write atomically so an interrupted download is never mistaken for complete.
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)
    return destination

def valid_zip(path):
    try:
        with ZipFile(path) as archive:
            return archive.testzip() is None
    except Exception:
        return False

download(REFERENCE_URL, REFERENCE_PATH)
download(SCORES_URL, SCORES_ZIP)

# Refuse to score corrupt or structurally incompatible inputs.
assert valid_zip(SCORES_ZIP), f"Corrupt archive: {SCORES_ZIP}"

if not any(SCORE_DIR.rglob("*.csv")):
    with ZipFile(SCORES_ZIP) as archive:
        archive.extractall(SCORE_DIR)

reference = pd.read_csv(REFERENCE_PATH)
required_reference_columns = {
    "DMS_id", "UniProt_ID", "coarse_selection_type",
}
assert required_reference_columns <= set(reference.columns)

score_files = {
    path.stem: path
    for path in SCORE_DIR.rglob("*.csv")
}
assert len(score_files) == 66, f"Expected 66 assays, found {len(score_files)}"

# A small schema check catches upstream archive changes early.
sample = pd.read_csv(next(iter(score_files.values())), nrows=2)
required_score_columns = {
    "mutated_sequence", "DMS_score", "PoET", *PUBLISHED_SIGNALS,
}
assert required_score_columns <= set(sample.columns)

print(f"reference assays: {len(reference)}")
print(f"published score files: {len(score_files)}")

# This is token confidence on an unmasked input, not masked pseudo-log-likelihood.
def score_unmasked_token_confidence(sequences, tokenizer, model, device):
    import torch

    scores = np.zeros(len(sequences), dtype=np.float64)
    # Similar-length batches minimize padding and GPU memory use.
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))

    start = 0
    while start < len(order):
        stop = start
        longest = len(sequences[order[start]])

        while stop < len(order):
            candidate_longest = max(
                longest,
                len(sequences[order[stop]]),
            )
            candidate_size = stop - start + 1
            if candidate_size * candidate_longest > ESM2_BATCH_TOKENS:
                break
            longest = candidate_longest
            stop += 1

        indices = order[start:max(stop, start + 1)]
        start = max(stop, start + 1)
        batch = [
            sequences[index][:ESM2_MAX_RESIDUES]
            for index in indices
        ]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            logits = model(**encoded).logits.float()
            log_probabilities = torch.log_softmax(logits, dim=-1)

        # Gather the log probability assigned to each observed residue.
        token_ids = encoded["input_ids"]
        observed_logp = log_probabilities.gather(
            -1,
            token_ids.unsqueeze(-1),
        ).squeeze(-1)

        # BOS, EOS, and padding do not contribute to the sequence score.
        special = torch.zeros_like(token_ids, dtype=torch.bool)
        for token_id in tokenizer.all_special_ids:
            special |= token_ids == token_id

        valid = encoded["attention_mask"].bool() & ~special
        batch_scores = (
            (observed_logp * valid).sum(axis=1) /
            valid.sum(axis=1).clamp(min=1)
        )

        for position, original_index in enumerate(indices):
            scores[original_index] = batch_scores[position].item()

        del encoded, logits, log_probabilities
        del token_ids, observed_logp, special, valid, batch_scores

    return scores

def discover_esm2_files():
    # Prefer the artifact cache, but reuse arrays from the original Colab run.
    files = {}
    for directory in ESM2_SEARCH_DIRS:
        if directory.exists():
            files.update({
                path.stem: path
                for path in directory.glob("*.npy")
            })
    return files

esm2_files = discover_esm2_files()

if GENERATE_ESM2 and len(esm2_files) < 65:
    import gc
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from tqdm.auto import tqdm

    assert torch.cuda.is_available(), "ESM2 generation requires a GPU runtime"
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
    model = (
        AutoModelForMaskedLM
        .from_pretrained(ESM2_MODEL)
        .half()
        .to(device)
        .eval()
    )

    # One file per assay makes this expensive pass resumable.
    for assay, path in tqdm(sorted(score_files.items())):
        destination = ESM2_DIR / f"{assay}.npy"
        if assay in SKIP_ESM2 or assay in esm2_files:
            continue

        sequences = (
            pd.read_csv(path, usecols=["mutated_sequence"])
            ["mutated_sequence"]
            .astype(str)
            .tolist()
        )
        values = score_unmasked_token_confidence(
            sequences,
            tokenizer,
            model,
            device,
        )
        np.save(destination, values)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

esm2_files = discover_esm2_files()
print(f"ESM2 arrays available: {len(esm2_files)}")
if len(esm2_files) < 65:
    searched = "\n  ".join(map(str, ESM2_SEARCH_DIRS))
    raise RuntimeError(
        f"Found {len(esm2_files)}/65 ESM2 arrays. "
        "Use a GPU to generate the missing arrays or mount/copy the saved cache. "
        f"Searched:\n  {searched}"
    )

assert len(esm2_files) >= 65, (
    "The frozen method requires the 65 ESM2 arrays. "
    "Generate them above or copy the saved arrays into ESM2_DIR."
)

# Metadata supplies the hierarchy and the protein-disjoint split.
metadata = (
    reference
    .assign(
        coarse_selection_type=lambda frame:
            frame["coarse_selection_type"].astype(str)
    )
    .set_index("DMS_id")
)

# Build every signal column from the same CSV row order.
built = []
for assay, path in sorted(score_files.items()):
    if assay not in metadata.index:
        continue

    frame = pd.read_csv(path)
    dms = frame["DMS_score"].to_numpy(dtype=float)
    signal_matrix = np.full(
        (len(frame), len(SIGNALS)),
        np.nan,
        dtype=np.float32,
    )

    for column_index, signal_name in enumerate(PUBLISHED_SIGNALS):
        values = pd.to_numeric(
            frame[signal_name],
            errors="coerce",
        ).to_numpy(dtype=float)
        if np.all(np.isfinite(values)):
            signal_matrix[:, column_index] = values

    # The saved ESM2 vector was generated from this exact sequence order.
    esm2_path = esm2_files.get(assay)
    if esm2_path is not None:
        esm2 = np.load(esm2_path)
        assert len(esm2) == len(frame), assay
        assert np.all(np.isfinite(esm2)), assay
        signal_matrix[:, -1] = esm2

    row = metadata.loc[assay]
    np.savez_compressed(
        CACHE_DIR / f"{assay}.npz",
        signals=signal_matrix,
        signal_names=np.asarray(SIGNALS),
        dms=dms,
        uniprot=str(row["UniProt_ID"]),
        selection=str(row["coarse_selection_type"]),
    )
    built.append(assay)

assert len(built) == 66

# Hold out entire proteins, never random variants from known proteins.
proteins = sorted({
    str(metadata.loc[assay, "UniProt_ID"])
    for assay in built
})
random.Random(SPLIT_SEED).shuffle(proteins)
heldout_proteins = set(proteins[:max(1, len(proteins) // 5)])

heldout_ids = {
    assay
    for assay in built
    if str(metadata.loc[assay, "UniProt_ID"]) in heldout_proteins
}
development_ids = set(built) - heldout_ids

assert development_ids.isdisjoint(heldout_ids)
assert development_ids | heldout_ids == set(built)

print(f"cache files: {len(built)}")
print(f"development assays: {len(development_ids)}")
print(f"held-out assays: {len(heldout_ids)}")

def rank_01(values):
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return np.full_like(values, 0.5)

    # Exact ordinal transform used by the reported method.
    return (
        np.argsort(np.argsort(values, kind="stable"), kind="stable") + 0.5
    ) / values.size

def prepare_ranks(cache):
    signals = np.asarray(cache["signals"], dtype=float)
    # Drop columns unavailable for the whole assay (ESM2 is absent once).
    present = ~np.all(np.isnan(signals), axis=0)
    names = np.asarray(cache["signal_names"]).astype(str)[present]
    signals = signals[:, present]

    ranks = np.column_stack([
        rank_01(signals[:, column])
        for column in range(signals.shape[1])
    ])
    # Collapse related published models separately from the orthogonal signal.
    is_esm2 = np.asarray(["ESM2" in name for name in names])
    return ranks, is_esm2, ~is_esm2

def cluster_votes(cache):
    ranks, is_esm2, is_published = prepare_ranks(cache)
    n_variants = ranks.shape[0]

    # Median gives the correlated crowd one robust vote.
    crowd = (
        np.median(ranks[:, is_published], axis=1)
        if is_published.any()
        else np.full(n_variants, 0.5)
    )
    esm2 = (
        np.mean(ranks[:, is_esm2], axis=1)
        if is_esm2.any()
        else np.full(n_variants, 0.5)
    )
    return ranks, crowd, esm2, is_published.any(), is_esm2.any()

def equal_weight(cache):
    ranks, _, _ = prepare_ranks(cache)
    return ranks.mean(axis=1)

def clean_full(cache):
    ranks, crowd, esm2, has_published, has_esm2 = cluster_votes(cache)

    # Geometric mean gives the crowd and ESM2 equal structural influence.
    if has_published and has_esm2:
        base = np.sqrt(
            np.clip(crowd, 1e-9, 1) *
            np.clip(esm2, 1e-9, 1)
        )
    else:
        base = crowd if has_published else esm2

    # Variants with stronger inter-model disagreement are discounted slightly.
    agreement = 1 / (1 + np.std(ranks, axis=1) ** 2)
    return np.nan_to_num(base * agreement, nan=0.0)

def clean_no_agreement(cache):
    _, crowd, esm2, has_published, has_esm2 = cluster_votes(cache)
    if has_published and has_esm2:
        return np.sqrt(
            np.clip(crowd, 1e-9, 1) *
            np.clip(esm2, 1e-9, 1)
        )
    return crowd if has_published else esm2

def clean_no_split(cache):
    ranks, _, _ = prepare_ranks(cache)
    geometric_mean = np.exp(
        np.mean(np.log(np.clip(ranks, 1e-9, 1)), axis=1)
    )
    agreement = 1 / (1 + np.std(ranks, axis=1) ** 2)
    return geometric_mean * agreement

METHODS = {
    "equal_weight": equal_weight,
    "clean_full": clean_full,
    "clean_no_agreement": clean_no_agreement,
    "clean_no_split": clean_no_split,
}

# Minimal guard for output shape, finiteness, and deterministic behavior.
demo = {
    "signals": np.array([
        [0.1, 0.2, 0.8],
        [0.4, 0.3, 0.5],
        [0.9, 0.7, 0.2],
        [0.6, 0.8, 0.6],
    ]),
    "signal_names": np.array([
        "Progen2_medium",
        "Tranception_M",
        "ESM2_650M",
    ]),
}

for name, method in METHODS.items():
    values = method(demo)
    assert values.shape == (4,), name
    assert np.all(np.isfinite(values)), name
    assert np.array_equal(values, method(demo)), name

print("combiner checks passed")

# Official hierarchy: assays -> protein/selection -> selection -> benchmark.
def official_aggregate(table, score_column):
    grouped = (
        table
        .dropna(subset=[score_column])
        .groupby(["selection", "uniprot"])[score_column]
        .mean()
        .reset_index()
    )
    return float(
        grouped
        .groupby("selection")[score_column]
        .mean()
        .mean()
    )

# Produce one auditable record per assay for every method.
rows = []
for cache_path in sorted(CACHE_DIR.glob("*.npz")):
    assay = cache_path.stem

    with np.load(cache_path, allow_pickle=True) as stored:
        cache = {
            "signals": stored["signals"],
            "signal_names": stored["signal_names"],
        }
        dms = np.asarray(stored["dms"], dtype=float)
        uniprot = str(stored["uniprot"])
        selection = str(stored["selection"])

    published = pd.read_csv(
        score_files[assay],
        usecols=["DMS_score", "PoET"],
    )
    # Verify that cached signals and published labels still share row order.
    alignment = spearmanr(
        published["DMS_score"].to_numpy(dtype=float),
        dms,
    ).correlation
    assert alignment > 0.999, f"Row mismatch: {assay}"

    record = {
        "DMS_id": assay,
        "uniprot": uniprot,
        "selection": selection,
        "split": (
            "heldout"
            if assay in heldout_ids
            else "development"
        ),
        "PoET": spearmanr(
            pd.to_numeric(published["PoET"], errors="coerce"),
            dms,
        ).correlation,
    }

    for method_name, method in METHODS.items():
        prediction = np.asarray(method(cache), dtype=float)
        assert prediction.shape == dms.shape, method_name
        assert np.all(np.isfinite(prediction)), method_name
        record[method_name] = spearmanr(prediction, dms).correlation

    rows.append(record)

assay_scores = pd.DataFrame(rows)
assert len(assay_scores) == 66

# Report full, development, and untouched-protein aggregates side by side.
summary_rows = []
for method_name in ["PoET", *METHODS]:
    summary_rows.append({
        "method": method_name,
        "full": official_aggregate(assay_scores, method_name),
        "development": official_aggregate(
            assay_scores[assay_scores["split"] == "development"],
            method_name,
        ),
        "heldout": official_aggregate(
            assay_scores[assay_scores["split"] == "heldout"],
            method_name,
        ),
    })

summary = pd.DataFrame(summary_rows).set_index("method")
summary["heldout_delta_vs_equal"] = (
    summary["heldout"] - summary.loc["equal_weight", "heldout"]
)
print("\nAggregate results")
print(summary.round(4).to_string())

def protein_cluster_bootstrap(
    table,
    candidate="clean_full",
    reference_method="PoET",
    repeats=N_BOOTSTRAPS,
    seed=BOOTSTRAP_SEED,
):
    columns = [
        "uniprot", "selection",
        candidate, reference_method,
    ]
    base = table[columns].dropna().reset_index(drop=True)
    proteins = base["uniprot"].unique()
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype=float)

    # Resample whole proteins so related assays move together.
    for repeat in range(repeats):
        sampled = rng.choice(
            proteins,
            size=len(proteins),
            replace=True,
        )

        chunks = []
        for replicate, protein in enumerate(sampled):
            chunk = base[base["uniprot"] == protein].copy()
            # A synthetic ID preserves multiplicity when one protein is sampled twice.
            chunk["bootstrap_protein"] = replicate
            chunks.append(chunk)

        bootstrap = pd.concat(chunks, ignore_index=True)
        # Recompute the benchmark hierarchy inside every bootstrap replicate.
        per_protein = (
            bootstrap
            .groupby(["selection", "bootstrap_protein"])[
                [candidate, reference_method]
            ]
            .mean()
        )
        aggregate = per_protein.groupby("selection").mean().mean()
        differences[repeat] = (
            aggregate[candidate] - aggregate[reference_method]
        )

    return pd.Series({
        "candidate": candidate,
        "reference": reference_method,
        "point_difference": (
            official_aggregate(table, candidate) -
            official_aggregate(table, reference_method)
        ),
        "bootstrap_mean": differences.mean(),
        "ci_2.5": np.percentile(differences, 2.5),
        "ci_97.5": np.percentile(differences, 97.5),
        "p_candidate_gt_reference": np.mean(differences > 0),
    }), differences

bootstrap_summary, bootstrap_differences = protein_cluster_bootstrap(
    assay_scores
)
print("\nProtein-cluster bootstrap")
print(bootstrap_summary.to_string())

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

# Persist granular scores, summary statistics, and the uncertainty distribution.
assay_scores.to_csv(RESULT_DIR / "assay_scores.csv", index=False)
summary.to_csv(RESULT_DIR / "summary.csv")
pd.DataFrame({
    "clean_full_minus_PoET": bootstrap_differences,
}).to_csv(
    RESULT_DIR / "bootstrap_clean_full_vs_PoET.csv",
    index=False,
)

# Hash source data and record every knob needed to audit this run.
manifest = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "reference_sha256": sha256(REFERENCE_PATH),
    "published_scores_zip_sha256": sha256(SCORES_ZIP),
    "esm2_model": ESM2_MODEL,
    "esm2_arrays": len(esm2_files),
    "signals": SIGNALS,
    "split_seed": SPLIT_SEED,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "n_bootstraps": N_BOOTSTRAPS,
    "rank_tie_policy": "stable ordinal; reproduces reported method",
    "esm2_feature": "one-pass unmasked token confidence",
}
(RESULT_DIR / "manifest.json").write_text(
    json.dumps(manifest, indent=2)
)

print("\nSaved results")
for path in sorted(RESULT_DIR.iterdir()):
    print(path.name)

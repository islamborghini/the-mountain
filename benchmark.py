"""Reproduce the frozen ProteinGym v1.3 indel ensemble; no import-time I/O."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path, PurePosixPath
import random
import tempfile
from zipfile import ZipFile

import numpy as np
import pandas as pd

from evaluate import evaluate_assay, official_aggregate, protein_cluster_bootstrap
from model import MODEL_VERSION, PUBLISHED_SIGNALS, score_variants

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
SKIP_ESM2 = {"CAPSD_AAV2S_Sinai_2021_designed_indels"}
PROFILES = {
    "original_24": [*PUBLISHED_SIGNALS, "ESM2_650M"],
    "sequence_6": ["RITA_s", "RITA_m", "RITA_l", "RITA_xl", "ProtGPT2", "ESM2_650M"],
}
METHODS = {"The_Mountain": "mountain", "equal_weight": "equal_weight", "no_agreement": "no_agreement", "no_split": "no_split"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_hash(sequences):
    return hashlib.sha256(json.dumps(list(sequences), ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, partial = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        writer(Path(partial))
        os.replace(partial, path)
    finally:
        Path(partial).unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, lambda partial: partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"))


def write_csv(path, frame):
    atomic_write(path, lambda partial: frame.to_csv(partial, index=False))


def validated_members(archive, suffix, expected_ids):
    """Never extract archives; reject path traversal, duplicate IDs and omissions."""
    members = {}
    for info in archive.infolist():
        name = PurePosixPath(info.filename)
        if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
            raise ValueError(f"Unsafe ZIP member: {info.filename}")
        if info.filename.startswith("__MACOSX/") or info.is_dir():
            continue
        if name.suffix != suffix:
            raise ValueError(f"Unexpected ZIP member: {info.filename}")
        if name.stem in members:
            raise ValueError(f"Duplicate assay in archive: {name.stem}")
        # Published files are modest; this also limits accidental decompression bombs.
        if info.file_size > 512 * 1024 * 1024:
            raise ValueError(f"Oversized ZIP member: {info.filename}")
        members[name.stem] = info.filename
    if set(members) != set(expected_ids):
        raise ValueError(f"Archive assay membership differs: missing={set(expected_ids)-set(members)}, extra={set(members)-set(expected_ids)}")
    if archive.testzip() is not None:
        raise ValueError("Corrupt ZIP archive")
    return members


def load_reference():
    manifest = json.loads((REFERENCE_DIR / "benchmark_manifest.json").read_text())
    reference_path = REFERENCE_DIR / "DMS_indels.csv"
    if sha256_file(reference_path) != manifest["reference_sha256"]:
        raise ValueError("Committed benchmark reference hash mismatch")
    reference = pd.read_csv(reference_path)
    if reference.DMS_id.duplicated().any() or set(reference.DMS_id) != set(manifest["assays"]):
        raise ValueError("Benchmark reference assay membership mismatch")
    return reference, manifest


def download_scores(path, manifest):
    if not path.exists():
        import requests
        def fetch(partial):
            with requests.get(manifest["scores_url"], stream=True, timeout=120) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if sha256_file(partial) != manifest["scores_archive_sha256"]:
                raise ValueError("Downloaded published archive hash mismatch")
        atomic_write(path, fetch)
    if sha256_file(path) != manifest["scores_archive_sha256"]:
        raise ValueError("Published archive hash mismatch; expected frozen v1.3 release")


def validate_score_frame(frame, raw, assay, spec):
    required = {"mutated_sequence", "DMS_score", "DMS_score_bin", "PoET", *PUBLISHED_SIGNALS}
    if not required.issubset(frame):
        raise ValueError(f"{assay}: score schema missing {required-set(frame)}")
    if hashlib.sha256(raw).hexdigest() != spec["score_csv_sha256"]:
        raise ValueError(f"{assay}: score CSV hash mismatch")
    if len(frame) != spec["rows"] or frame.mutated_sequence.isna().any():
        raise ValueError(f"{assay}: invalid row count or empty sequence")
    if sequence_hash(frame.mutated_sequence.astype(str)) != spec["ordered_sequences_sha256"]:
        raise ValueError(f"{assay}: sequence alignment mismatch")
    if not np.isfinite(pd.to_numeric(frame.DMS_score, errors="raise").to_numpy(float)).all():
        raise ValueError(f"{assay}: experimental labels must be finite")


def cache_esm2(path, values, sequences, provenance):
    values = np.asarray(values, dtype=np.float32)
    atomic_write(path, lambda partial: _save_npy(partial, values))
    write_json(path.with_suffix(".json"), {
        "schema_version": 1, "signal": "ESM2_650M", "rows": len(sequences),
        "ordered_sequences_sha256": sequence_hash(sequences),
        "array_sha256": sha256_file(path), "provenance": provenance,
    })


def _save_npy(path, values):
    with path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)


def read_esm2(path, sequences):
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise ValueError(f"Unbound ESM2 cache {path}; arrays require sequence-hash metadata")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("signal") != "ESM2_650M" or metadata.get("ordered_sequences_sha256") != sequence_hash(sequences):
        raise ValueError(f"ESM2 cache sequence mismatch: {path}")
    if metadata.get("array_sha256") != sha256_file(path):
        raise ValueError(f"ESM2 cache content hash mismatch: {path}")
    values = np.load(path, allow_pickle=False)
    if values.shape != (len(sequences),) or not np.isfinite(values).all():
        raise ValueError(f"Invalid ESM2 vector: {path}")
    return values, metadata


def migrate_legacy(archive, member, frame, assay, destination, legacy_hash):
    """Explicit historical migration, not retrospective proof of ESM2 provenance."""
    with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as stored:
        names = stored["signal_names"].astype(str).tolist()
        signals = stored["signals"]
        if names != [*PUBLISHED_SIGNALS, "ESM2_650M"] or signals.shape != (len(frame), len(names)):
            raise ValueError(f"{assay}: legacy signal schema differs")
        if not np.allclose(stored["dms"], frame.DMS_score.to_numpy(float), rtol=1e-12, atol=1e-12):
            raise ValueError(f"{assay}: legacy experimental row alignment differs")
        # Check each component with only float32 parser-rounding tolerance.
        # Label equality at decimal parse tolerance independently anchors row order.
        for i, name in enumerate(PUBLISHED_SIGNALS):
            actual = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float32)
            if np.isfinite(actual).all():
                if not np.allclose(signals[:, i], actual, rtol=2 * np.finfo(np.float32).eps, atol=np.finfo(np.float32).tiny):
                    raise ValueError(f"{assay}: legacy component row alignment differs ({name})")
            elif not np.isnan(signals[:, i]).all():
                raise ValueError(f"{assay}: legacy missing component policy differs ({name})")
        values = np.asarray(signals[:, -1], dtype=np.float32)
    if assay in SKIP_ESM2:
        if not np.isnan(values).all():
            raise ValueError(f"{assay}: expected historical consensus-only fallback")
        return
    if not np.isfinite(values).all():
        raise ValueError(f"{assay}: legacy ESM2 must be finite")
    cache_esm2(destination, values, frame.mutated_sequence.astype(str).tolist(), {
        "kind": "explicit_legacy_migration", "archive_sha256": legacy_hash,
        "alignment_checks": "labels within 1e-12 decimal parsing tolerance and components within two float32 rounding units",
        "limitation": "Legacy archive did not store sequences/checkpoint metadata. Historical ESM2 ordering and weights cannot be independently established; new hash binding is prospective.",
        "historical_scoring_policy": "unmasked token confidence, prefix truncation at 1022 residues",
    })


def generate_benchmark_components(published, members, manifest, root, esm2_dir, device):
    """Score each checkpoint once over all assays; keep only small score arrays.

    Labels are excluded before passing sequences to the component scorer.
    Model scoring internally batches sequences; matrices are split back into
    assay order before label-free prediction.
    """
    from component_scoring import generate_signals
    all_sequences, bounds, sequences_by_assay = [], {}, {}
    for assay, spec in manifest["assays"].items():
        raw = published.read(members[assay])
        frame = pd.read_csv(io.BytesIO(raw))
        validate_score_frame(frame, raw, assay, spec)
        sequences = frame.mutated_sequence.astype(str).tolist()
        start = len(all_sequences)
        all_sequences.extend(sequences)
        bounds[assay] = (start, len(all_sequences))
        sequences_by_assay[assay] = sequences
    causal_names = PROFILES["sequence_6"][:-1]
    matrix, returned_names, causal_metadata = generate_signals(
        all_sequences, signal_names=causal_names, device=device,
        cache_dir=Path(root) / "cache" / "components", length_policy="truncate")
    if list(returned_names) != causal_names or matrix.shape != (len(all_sequences), len(causal_names)) or not np.isfinite(matrix).all():
        raise ValueError("Native causal scoring returned incompatible shape/names")
    native = {assay: np.asarray(matrix[start:end], dtype=np.float32).copy()
              for assay, (start, end) in bounds.items()}
    del matrix, all_sequences
    esm_sequences, esm_bounds = [], {}
    for assay, sequences in sequences_by_assay.items():
        if assay not in SKIP_ESM2:
            start = len(esm_sequences)
            esm_sequences.extend(sequences)
            esm_bounds[assay] = (start, len(esm_sequences))
    esm_matrix, returned_names, esm_metadata = generate_signals(
        esm_sequences, signal_names=["ESM2_650M"], device=device,
        cache_dir=Path(root) / "cache" / "components", length_policy="truncate")
    if list(returned_names) != ["ESM2_650M"] or esm_matrix.shape != (len(esm_sequences), 1) or not np.isfinite(esm_matrix).all():
        raise ValueError("Native ESM2 scoring returned incompatible shape/names")
    for assay, (start, end) in esm_bounds.items():
        cache_esm2(esm2_dir / f"{assay}.npy", esm_matrix[start:end, 0],
                   sequences_by_assay[assay], {"kind": "generated", "features": esm_metadata,
                   "generation_group": "full_benchmark", "group_slice": [start, end]})
    return native, causal_metadata


def run_benchmark(root, *, scores_zip=None, esm2_dir=None, legacy_cache_zip=None,
                  generate_esm2=False, generate_components=False, profile="original_24", bootstrap_repeats=5000,
                  bootstrap_seed=0, device="auto"):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile}")
    if generate_components and profile != "sequence_6":
        raise ValueError("--generate-components supports only the native sequence_6 profile")
    if generate_components and legacy_cache_zip:
        raise ValueError("--generate-components cannot be combined with historical --legacy-cache-zip")
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    root = Path(root)
    reference, manifest = load_reference()
    archive_path = Path(scores_zip) if scores_zip else root / "data" / "zero_shot_indels_scores.zip"
    download_scores(archive_path, manifest)
    esm2_dir = Path(esm2_dir) if esm2_dir else root / "data" / "esm2_650m"
    result_dir = root / "results"
    # A failed rerun must not leave a prior completion manifest attached to new partial files.
    (result_dir / "run_manifest.json").unlink(missing_ok=True)
    metadata = reference.set_index("DMS_id")
    proteins = sorted(reference.UniProt_ID.unique().tolist())
    random.Random(0).shuffle(proteins)
    historical_test = set(proteins[:max(1, len(proteins)//5)])
    legacy = None
    if legacy_cache_zip:
        legacy_hash = sha256_file(legacy_cache_zip)
        if legacy_hash != manifest["legacy_archive_sha256"]:
            raise ValueError("Legacy archive hash mismatch")
        legacy = ZipFile(legacy_cache_zip)
        legacy_members = validated_members(legacy, ".npz", manifest["assays"])
    rows, esm_provenance = [], {}
    try:
        with ZipFile(archive_path) as published:
            members = validated_members(published, ".csv", manifest["assays"])
            native_components, native_metadata = ({}, None)
            if generate_components:
                native_components, native_metadata = generate_benchmark_components(
                    published, members, manifest, root, esm2_dir, device)
            for assay, spec in manifest["assays"].items():
                raw = published.read(members[assay])
                frame = pd.read_csv(io.BytesIO(raw))
                validate_score_frame(frame, raw, assay, spec)
                sequences = frame.mutated_sequence.astype(str).tolist()
                esm_path = esm2_dir / f"{assay}.npy"
                if legacy is not None:
                    migrate_legacy(legacy, legacy_members[assay], frame, assay, esm_path, legacy_hash)
                if assay in SKIP_ESM2:
                    esm = np.full(len(frame), np.nan, dtype=np.float32)
                    esm_provenance[assay] = {"kind": "historical_consensus_only_fallback"}
                else:
                    if not esm_path.exists() and generate_esm2:
                        from component_scoring import generate_signals
                        matrix, _, feature_metadata = generate_signals(sequences, signal_names=["ESM2_650M"], device=device, cache_dir=root / "cache" / "components", length_policy="truncate")
                        cache_esm2(esm_path, matrix[:, 0], sequences, {"kind": "generated", "features": feature_metadata})
                    if not esm_path.exists():
                        raise ValueError(f"Missing {esm_path}; provide bound cache, --legacy-cache-zip or --generate-esm2")
                    esm, bound_metadata = read_esm2(esm_path, sequences)
                    esm_provenance[assay] = bound_metadata["provenance"]
                signals = np.full((len(frame), len(PROFILES[profile])), np.nan, dtype=np.float32)
                for i, name in enumerate(PROFILES[profile]):
                    if name == "ESM2_650M":
                        values = esm
                    elif generate_components:
                        values = native_components[assay][:, PROFILES["sequence_6"].index(name)]
                    else:
                        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(np.float32)
                    if np.isfinite(values).all():
                        signals[:, i] = values
                ids = [f"{assay}:{i}" for i in range(len(frame))]
                predictions = pd.DataFrame({"variant_id": ids, "mutated_sequence": sequences})
                for column, method in METHODS.items():
                    predictions[column] = score_variants(signals, PROFILES[profile], method=method)
                labels = pd.DataFrame({"variant_id": ids, "mutated_sequence": sequences, "DMS_score": frame.DMS_score.to_numpy(float)})
                record = {"DMS_id": assay, "uniprot": str(metadata.loc[assay, "UniProt_ID"]), "selection": str(metadata.loc[assay, "coarse_selection_type"]), "split": "historical_test" if str(metadata.loc[assay, "UniProt_ID"]) in historical_test else "historical_development", "variants": len(frame)}
                for column in METHODS:
                    record[column] = evaluate_assay(predictions, labels, prediction_column=column)
                baseline = predictions[["variant_id", "mutated_sequence"]].copy()
                baseline["PoET"] = frame.PoET.to_numpy(float)
                record["PoET"] = evaluate_assay(baseline, labels, prediction_column="PoET")
                rows.append(record)
                write_csv(result_dir / "predictions" / f"{assay}.csv", predictions)
                write_csv(result_dir / "labels" / f"{assay}.csv", labels)
                # Official script expects labels and baseline columns in merged files;
                # these are separate from the label-free prediction product.
                merged = frame.copy()
                merged["The_Mountain"] = predictions.The_Mountain.to_numpy()
                write_csv(result_dir / "official_merged" / f"{assay}.csv", merged)
        table = pd.DataFrame(rows)
        summary = pd.DataFrame([{"method": name, "full_unrounded": official_aggregate(table, name), "full_official_rounded": official_aggregate(table, name, round_assays=True), **{split: official_aggregate(table[table.split == split], name) for split in ("historical_development", "historical_test")}} for name in [*METHODS, "PoET"]])
        bootstrap, differences = protein_cluster_bootstrap(table, repeats=bootstrap_repeats, seed=bootstrap_seed)
        write_csv(result_dir / "assay_scores.csv", table)
        write_csv(result_dir / "summary.csv", summary)
        write_json(result_dir / "bootstrap.json", bootstrap)
        atomic_write(result_dir / "bootstrap_differences.npy", lambda partial: _save_npy(partial, differences))
        write_json(result_dir / "run_manifest.json", {
            "schema_version": 1, "profile": profile, "signals": PROFILES[profile],
            "model_version": MODEL_VERSION, "python": platform.python_version(),
            "dependencies": {name: version(name) for name in ["numpy", "pandas", "scipy"]},
            "tie_policy": "average", "signal_storage_dtype": "float32",
            "causal_feature_source": "native_checkpoint_scoring" if generate_components else "published_benchmark_scores",
            "native_causal_feature_metadata": native_metadata,
            "reference_sha256": manifest["reference_sha256"], "scores_archive_sha256": sha256_file(archive_path),
            "upstream_evaluator_commit": manifest["upstream_evaluator_commit"],
            "bootstrap_seed": bootstrap_seed, "bootstrap_repeats": bootstrap_repeats,
            "split_seed": 0, "validation_limitation": "Historical split was used during discovery; it is not untouched independent validation.",
            "esm2_provenance": esm_provenance,
            "outputs_sha256": {str(path.relative_to(result_dir)): sha256_file(path) for path in sorted(result_dir.rglob("*")) if path.is_file() and path.name != "run_manifest.json"},
        })
        print(summary.to_string(index=False))
        return table, summary
    finally:
        if legacy is not None:
            legacy.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("proteingym_artifact"))
    parser.add_argument("--scores-zip", type=Path)
    parser.add_argument("--esm2-dir", type=Path)
    parser.add_argument("--legacy-cache-zip", type=Path, help="Explicitly migrate pinned historical arrays; provenance limitations recorded")
    parser.add_argument("--generate-esm2", action="store_true")
    parser.add_argument("--generate-components", action="store_true", help="Generate every sequence_6 component from raw sequences (GPU recommended)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--profile", choices=list(PROFILES), default="original_24")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    run_benchmark(**vars(args))


if __name__ == "__main__":
    main()

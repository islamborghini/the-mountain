"""CSV prediction entry point; experimental labels never enter inference."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
import pandas as pd
import scipy

from model import MODEL_VERSION, SEQUENCE_SIGNALS, SIGNALS, score_variants


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_input(path, raw=False):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        for line, row in enumerate(reader, start=2):
            if row and len(row) != len(header):
                raise ValueError(f"CSV row {line} has {len(row)} fields; expected {len(header)}")
    if len(set(header)) != len(header):
        raise ValueError("CSV column names must be unique")
    frame = pd.read_csv(
        path, dtype={"protein_id": str, "variant_id": str, "mutated_sequence": str},
        keep_default_na=False,
        na_values={name: ["", "NA", "NaN", "nan", "N/A", "null"] for name in SIGNALS},
    )
    if "DMS_score" in frame or "PoET" in frame:
        raise ValueError("Prediction inputs must not contain DMS_score or PoET; remove experimental labels and comparison baselines")
    required = ["protein_id", "variant_id"] + (["mutated_sequence"] if raw else [])
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")
    if frame.empty:
        raise ValueError("Input CSV must contain at least one variant")
    for name in required:
        if frame[name].isna().any() or frame[name].str.strip().eq("").any():
            raise ValueError(f"Column {name} must contain a nonempty value for every variant")
    if frame.duplicated(["protein_id", "variant_id"]).any():
        raise ValueError("variant_id must be unique within each protein_id")
    identifiers = {"protein_id", "variant_id", "mutated_sequence"}
    extra = [name for name in frame if name not in identifiers and (raw or name not in SIGNALS)]
    if extra:
        raise ValueError(f"Unexpected input columns: {extra}")
    return frame


def predict_frame(frame, matrix, names, source="signals"):
    predictions = np.empty(len(frame), dtype=float)
    groups = []
    for protein, indices in frame.groupby("protein_id", sort=False).indices.items():
        values = matrix[indices]
        predictions[indices] = score_variants(values, names)
        available = [name for name, column in zip(names, values.T) if not np.isnan(column).all()]
        full = set(available) == set(SIGNALS)
        sequence_profile = set(available) == set(SEQUENCE_SIGNALS)
        groups.append({
            "protein_id": str(protein), "variants": len(indices), "signals": available,
            "profile": "original_24" if full else ("sequence_6" if sequence_profile else ("partial_sequence_ensemble" if source == "sequences" else "partial_scores_ensemble")),
            "original_full_profile": full,
        })
    identifiers = [name for name in ["protein_id", "variant_id", "mutated_sequence"] if name in frame]
    output = frame[identifiers].copy()
    output["mountain_score"] = predictions
    return output, groups


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--signals", type=Path, help="CSV of protein_id, variant_id and named component scores")
    source.add_argument("--sequences", type=Path, help="CSV of protein_id, variant_id and mutated_sequence; runs supported checkpoints")
    parser.add_argument("--output", type=Path, required=True, help="Prediction CSV; also writes <output>.manifest.json")
    parser.add_argument("--models", help="Comma-separated supported checkpoints for --sequences (default: " + ",".join(SEQUENCE_SIGNALS) + ")")
    parser.add_argument("--device", default="auto", help="Component-scoring device, e.g. auto, cpu, cuda, mps")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "the-mountain")
    parser.add_argument("--length-policy", choices=["error", "truncate"], default="error", help="ESM2 policy for sequences over 1022 residues; truncate explicitly uses the first 1022")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    raw = args.sequences is not None
    input_path = args.sequences if raw else args.signals
    manifest_path = Path(str(args.output) + ".manifest.json")
    try:
        if input_path.resolve() in {args.output.resolve(), manifest_path.resolve()}:
            raise ValueError("Output paths must differ from the input CSV")
        if args.models is not None and not raw:
            raise ValueError("--models applies only to --sequences")
        input_hash = sha256(input_path)
        frame = read_input(input_path, raw=raw)
        component_metadata = None
        if raw:
            from component_scoring import generate_signals

            requested = [name.strip() for name in args.models.split(",")] if args.models is not None else None
            if requested is not None and (not all(requested) or len(set(requested)) != len(requested)):
                raise ValueError("--models must contain unique nonempty model names")
            matrix, names, component_metadata = generate_signals(
                frame["mutated_sequence"].tolist(), signal_names=requested,
                device=args.device, cache_dir=args.cache_dir,
                length_policy=args.length_policy,
            )
            matrix = np.asarray(matrix, dtype=float)
            names = list(names)
        else:
            names = [name for name in SIGNALS if name in frame]
            matrix = frame[names].to_numpy(dtype=float)
        if matrix.ndim != 2 or matrix.shape != (len(frame), len(names)):
            raise ValueError("Component score matrix does not align with input variants")
        output, groups = predict_frame(frame, matrix, names, "sequences" if raw else "signals")
        if sha256(input_path) != input_hash:
            raise ValueError("Input CSV changed while predictions were being generated; rerun with a stable input")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.is_dir() or manifest_path.is_dir():
            raise ValueError("Output CSV and manifest destinations must be files")
        manifest = {
            "model_version": MODEL_VERSION, "method": "mountain",
            "source": "sequences" if raw else "signals",
            "input_sha256": input_hash,
            "rank_tie_policy": "average", "ranking_scope": "variants within each protein_id",
            "score_direction": "higher predicts greater fitness", "groups": groups,
            "component_scoring": component_metadata,
            "benchmark_record_verified": False,
            "benchmark_note": "This prediction run does not establish a record. Subset profiles require separate validation.",
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        }
        temporary = []
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv.tmp", dir=args.output.parent, delete=False) as handle:
                csv_temporary = Path(handle.name)
                temporary.append(csv_temporary)
                output.to_csv(handle, index=False)
            manifest["output_sha256"] = sha256(csv_temporary)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json.tmp", dir=args.output.parent, delete=False) as handle:
                manifest_temporary = Path(handle.name)
                temporary.append(manifest_temporary)
                handle.write(json.dumps(manifest, indent=2) + "\n")
            csv_temporary.replace(args.output)
            manifest_temporary.replace(manifest_path)
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    profiles = sorted({group["profile"] for group in groups})
    print(f"Wrote {len(output)} predictions to {args.output}; profiles: {', '.join(profiles)}")
    print(f"Provenance: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

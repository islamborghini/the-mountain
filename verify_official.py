"""Run the unmodified, hash-pinned ProteinGym indel evaluator on a release."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import requests

from benchmark import atomic_write, load_reference, sha256_file, write_json

UPSTREAM_COMMIT = "144fe22b07dfaeec2b366f2346203a9838a55b4c"
UPSTREAM_FILES = {
    "performance_DMS_benchmarks.py": "0002bc1b031c02dc2d67fde53da092c8fd301415645e8e83b425b4183570eba2",
    "constants.json": "e7b53a99952067b8c378d9072d38d71572b624fa357251bb8d3f53bf9b073e7a",
}


def frozen_evaluator(destination):
    destination = Path(destination)
    for name, expected in UPSTREAM_FILES.items():
        path = destination / name
        if not path.exists():
            url = f"https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/{UPSTREAM_COMMIT}/proteingym/{name}"
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            atomic_write(path, lambda temporary: temporary.write_bytes(response.content))
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen evaluator hash mismatch: {path}")
    return destination / "performance_DMS_benchmarks.py"


def verify_inputs(result_dir, reference, specification):
    manifest = json.loads((result_dir / "run_manifest.json").read_text())
    if manifest["reference_sha256"] != specification["reference_sha256"]:
        raise ValueError("Benchmark run uses a different reference snapshot")
    merged = result_dir / "official_merged"
    expected = set(reference.DMS_filename)
    if {path.name for path in merged.glob("*.csv")} != expected:
        raise ValueError("Official evaluator input must contain exactly all 66 assay files")
    for name in expected:
        relative = f"official_merged/{name}"
        if sha256_file(result_dir / relative) != manifest["outputs_sha256"][relative]:
            raise ValueError(f"Benchmark output was modified after generation: {relative}")
    for relative in ["assay_scores.csv", "summary.csv"]:
        if sha256_file(result_dir / relative) != manifest["outputs_sha256"][relative]:
            raise ValueError(f"Benchmark output was modified after generation: {relative}")
    return manifest


def compare_results(result_dir, official_dir):
    local = pd.read_csv(result_dir / "assay_scores.csv").set_index("DMS_id")
    official = pd.read_csv(official_dir / "Spearman" / "DMS_indels_Spearman_DMS_level.csv").set_index("DMS ID")
    if set(local.index) != set(official.index):
        raise ValueError("Official evaluator skipped or added assays")
    comparisons = {}
    for name, official_name in [("The_Mountain", "The_Mountain"), ("PoET", "PoET (200M)")]:
        expected = local[name].round(3).to_numpy()
        actual = official.loc[local.index, official_name].to_numpy(float)
        if not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=0, atol=1e-12):
            raise ValueError(f"Official per-assay results differ for {name}")
        comparisons[name] = {"assays": len(actual), "per_assay_match": True}
    summary = pd.read_csv(official_dir / "Spearman" / "Summary_performance_DMS_indels_Spearman.csv").set_index("Model_name")
    expected_summary = pd.read_csv(result_dir / "summary.csv").set_index("method")
    for name, official_name in [("The_Mountain", "The_Mountain"), ("PoET", "PoET (200M)")]:
        actual = float(summary.loc[official_name, "Average_Spearman"])
        expected = round(float(expected_summary.loc[name, "full_official_rounded"]), 3)
        if not np.isfinite(actual) or abs(actual - expected) > 1e-12:
            raise ValueError(f"Official aggregate differs for {name}: {actual} vs {expected}")
        comparisons[name]["official_spearman"] = actual
    return comparisons


def run_verification(root, output_dir=None):
    if importlib.util.find_spec("sklearn") is None:
        raise RuntimeError("Install the evaluator dependency: python -m pip install '.[verification]'")
    root = Path(root).resolve()
    result_dir = root / "results"
    reference, specification = load_reference()
    manifest = verify_inputs(result_dir, reference, specification)
    script = frozen_evaluator(root / "cache" / "official_evaluator" / UPSTREAM_COMMIT / "proteingym")
    official_dir = Path(output_dir).resolve() if output_dir else result_dir / "official_evaluation"
    official_dir.mkdir(parents=True, exist_ok=True)
    config_path = official_dir / "config.json"
    write_json(config_path, {"model_list_zero_shot_indels_DMS": {
        "The_Mountain": {"model_type": "Ensemble"},
        "PoET": {"model_type": "Hybrid - Alignment & PLM"},
    }})
    # The unmodified evaluator reads the DMS data only to count rows. The frozen
    # merged input contains all original labels and sequences, so it serves both.
    reference_path = result_dir / "official_reference.csv"
    from benchmark import REFERENCE_DIR
    atomic_write(reference_path, lambda temporary: temporary.write_bytes((REFERENCE_DIR / "DMS_indels.csv").read_bytes()))
    command = [sys.executable, str(script),
               "--input_scoring_files_folder", str(result_dir / "official_merged"),
               "--output_performance_file_folder", str(official_dir),
               "--DMS_reference_file_path", str(reference_path),
               "--DMS_data_folder", str(result_dir / "official_merged"),
               "--config_file", str(config_path), "--indel_mode"]
    completed = subprocess.run(command, capture_output=True, text=True)
    atomic_write(official_dir / "evaluator.log", lambda temporary: temporary.write_text(completed.stdout + completed.stderr))
    if completed.returncode:
        raise RuntimeError(f"Official evaluator failed; see {official_dir / 'evaluator.log'}")
    comparisons = compare_results(result_dir, official_dir)
    report = {
        "upstream_commit": UPSTREAM_COMMIT, "upstream_sha256": UPSTREAM_FILES,
        "profile": manifest["profile"], "reference_sha256": specification["reference_sha256"],
        "run_manifest_sha256": sha256_file(result_dir / "run_manifest.json"),
        "comparisons": comparisons,
        "exceeds_PoET_point_estimate": comparisons["The_Mountain"]["official_spearman"] > comparisons["PoET"]["official_spearman"],
        "limitation": "Validates historical benchmark scores and evaluator agreement; not a fresh component-inference run or official leaderboard acceptance.",
    }
    write_json(official_dir / "verification.json", report)
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("proteingym_artifact"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        run_verification(args.root, args.output_dir)
    except (ValueError, OSError, RuntimeError, KeyError, requests.RequestException) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())

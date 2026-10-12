import hashlib
import json

import pandas as pd
import pytest

import verify_official as verification


def test_frozen_evaluator_rejects_modified_code_before_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(verification, "UPSTREAM_FILES", {"evaluator.py": hashlib.sha256(b"expected").hexdigest()})
    (tmp_path / "evaluator.py").write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        verification.frozen_evaluator(tmp_path)


def test_verification_requires_complete_unmodified_assay_exports(tmp_path):
    merged = tmp_path / "official_merged"
    merged.mkdir()
    files = {"official_merged/a.csv": b"scores", "assay_scores.csv": b"assays", "summary.csv": b"summary"}
    for name, contents in files.items():
        (tmp_path / name).write_bytes(contents)
    manifest = {"reference_sha256": "reference", "outputs_sha256": {
        name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()}}
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    reference = pd.DataFrame({"DMS_filename": ["a.csv"]})
    verification.verify_inputs(tmp_path, reference, {"reference_sha256": "reference"})
    (merged / "a.csv").write_text("tampered")
    with pytest.raises(ValueError, match="modified"):
        verification.verify_inputs(tmp_path, reference, {"reference_sha256": "reference"})
    with pytest.raises(ValueError, match="exactly all"):
        verification.verify_inputs(tmp_path, pd.DataFrame({"DMS_filename": ["a.csv", "missing.csv"]}), {"reference_sha256": "reference"})


def test_official_comparison_detects_skipped_and_wrong_scores(tmp_path):
    local = tmp_path / "local"
    official = tmp_path / "official" / "Spearman"
    local.mkdir()
    official.mkdir(parents=True)
    pd.DataFrame({"DMS_id": ["a", "b"], "The_Mountain": [.5576, .5584], "PoET": [.5166, .5166]}).to_csv(local / "assay_scores.csv", index=False)
    pd.DataFrame({"method": ["The_Mountain", "PoET"], "full_official_rounded": [.558, .517]}).to_csv(local / "summary.csv", index=False)
    assay_path = official / "DMS_indels_Spearman_DMS_level.csv"
    frame = pd.DataFrame({"DMS ID": ["a", "b"], "The_Mountain": [.558, .558], "PoET (200M)": [.517, .517]})
    frame.to_csv(assay_path, index=False)
    pd.DataFrame({"Model_name": ["The_Mountain", "PoET (200M)"], "Average_Spearman": [.558, .517]}).to_csv(official / "Summary_performance_DMS_indels_Spearman.csv", index=False)
    assert verification.compare_results(local, official.parent)["The_Mountain"]["per_assay_match"]
    frame.iloc[:1].to_csv(assay_path, index=False)
    with pytest.raises(ValueError, match="skipped"):
        verification.compare_results(local, official.parent)
    frame.loc[0, "The_Mountain"] = .1
    frame.to_csv(assay_path, index=False)
    with pytest.raises(ValueError, match="per-assay"):
        verification.compare_results(local, official.parent)

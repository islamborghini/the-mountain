import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from model import SEQUENCE_SIGNALS, SIGNALS
from predict import main, predict_frame, read_input


def test_score_cli_preserves_ids_and_groups(tmp_path):
    source = tmp_path / "signals.csv"
    output = tmp_path / "predictions.csv"
    pd.DataFrame({"protein_id": ["001", "002", "001", "002"], "variant_id": ["01", "01", "02", "02"], "RITA_s": [1, 2, 3, 1]}).to_csv(source, index=False)
    assert main(["--signals", str(source), "--output", str(output)]) == 0
    result = pd.read_csv(output, dtype=str)
    assert result["protein_id"].tolist() == ["001", "002", "001", "002"]
    assert result["variant_id"].tolist() == ["01", "01", "02", "02"]
    np.testing.assert_allclose(result["mountain_score"].astype(float), [0.25, 0.75, 0.75, 0.25])
    manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text())
    assert manifest["rank_tie_policy"] == "average"
    assert manifest["groups"][0]["profile"] == "partial_scores_ensemble"
    assert manifest["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["benchmark_record_verified"] is False


@pytest.mark.parametrize("extra,message", [("DMS_score", "experimental labels"), ("PoET", "comparison baselines"), ("unknown", "Unexpected")])
def test_cli_rejects_labels_baselines_unknown_columns(tmp_path, extra, message):
    source = tmp_path / "input.csv"
    source.write_text(f"protein_id,variant_id,RITA_s,{extra}\np,v,1,3\n")
    with pytest.raises(ValueError, match=message):
        read_input(source)


def test_duplicate_identifiers_and_headers_are_rejected(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("protein_id,variant_id,RITA_s\np,v,1\np,v,2\n")
    with pytest.raises(ValueError, match="unique within"):
        read_input(source)
    source.write_text("protein_id,variant_id,RITA_s,RITA_s\np,v,1,2\n")
    with pytest.raises(ValueError, match="column names must be unique"):
        read_input(source)


def test_literal_na_ids_are_preserved_and_empty_signals_are_missing(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("protein_id,variant_id,RITA_s,ESM2_650M\nNA,NA,1,\nNA,v2,2,\n")
    frame = read_input(source)
    assert frame["protein_id"].tolist() == ["NA", "NA"]
    assert frame["variant_id"].tolist() == ["NA", "v2"]
    assert frame["ESM2_650M"].isna().all()


@pytest.mark.parametrize("row", ["p,v,1,2", "p,v"])
def test_malformed_csv_rows_do_not_shift_or_omit_scores(tmp_path, row):
    source = tmp_path / "input.csv"
    source.write_text("protein_id,variant_id,RITA_s\n" + row + "\n")
    with pytest.raises(ValueError, match="fields; expected 3"):
        read_input(source)


def test_raw_sequence_cli_lazily_calls_runtime(tmp_path, monkeypatch):
    source = tmp_path / "sequences.csv"
    source.write_text("protein_id,variant_id,mutated_sequence\np,original,ACDE\np,deletion,ACE\n")
    output = tmp_path / "predictions.csv"
    calls = []

    def generate(sequences, **kwargs):
        calls.append((sequences, kwargs))
        return np.array([[1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5]]), SEQUENCE_SIGNALS, {"test_runtime": True}

    monkeypatch.setitem(sys.modules, "component_scoring", SimpleNamespace(generate_signals=generate))
    assert main(["--sequences", str(source), "--output", str(output), "--device", "cpu", "--cache-dir", str(tmp_path / "cache")]) == 0
    assert calls[0][0] == ["ACDE", "ACE"]
    assert calls[0][1]["signal_names"] is None
    assert calls[0][1]["device"] == "cpu"
    assert calls[0][1]["length_policy"] == "error"
    manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text())
    assert manifest["groups"][0]["profile"] == "sequence_6"
    assert manifest["component_scoring"] == {"test_runtime": True}
    assert pd.read_csv(output)["mutated_sequence"].tolist() == ["ACDE", "ACE"]


def test_profile_labels_include_unavailable_columns():
    frame = pd.DataFrame({"protein_id": ["p", "p"], "variant_id": ["v1", "v2"]})
    matrix = np.tile(np.array([1., 2.])[:, None], (1, len(SIGNALS)))
    assert predict_frame(frame, matrix, SIGNALS)[1][0]["profile"] == "original_24"
    matrix[:, -1] = np.nan
    assert predict_frame(frame, matrix, SIGNALS)[1][0]["profile"] == "partial_scores_ensemble"


def test_input_cannot_be_overwritten(tmp_path):
    source = tmp_path / "input.csv"
    contents = "protein_id,variant_id,RITA_s\np,v,1\n"
    source.write_text(contents)
    with pytest.raises(SystemExit):
        main(["--signals", str(source), "--output", str(source)])
    assert source.read_text() == contents


def test_failed_output_write_preserves_previous_release(tmp_path, monkeypatch):
    source = tmp_path / "input.csv"
    source.write_text("protein_id,variant_id,RITA_s\np,v,1\n")
    output = tmp_path / "predictions.csv"
    output.write_text("previous predictions\n")
    manifest = Path(str(output) + ".manifest.json")
    manifest.write_text("previous manifest\n")

    def fail_write(*args, **kwargs):
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_write)
    with pytest.raises(SystemExit):
        main(["--signals", str(source), "--output", str(output)])
    assert output.read_text() == "previous predictions\n"
    assert manifest.read_text() == "previous manifest\n"
    assert not list(tmp_path.glob("*.tmp"))

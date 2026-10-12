"""Evaluation only: experimental labels never enter the ensemble model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def official_aggregate(table: pd.DataFrame, column: str, *, round_assays=False) -> float:
    """Equal weight assays within protein/category, proteins, then categories.

    ProteinGym's evaluator rounds assay metrics to 3 decimals before aggregation.
    Keep unrounded results as well to avoid hiding small changes.
    """
    working = table[["uniprot", "selection", column]].copy()
    if round_assays:
        working[column] = working[column].round(3)
    grouped = working.dropna(subset=[column]).groupby(["selection", "uniprot"])[column].mean()
    if grouped.empty:
        raise ValueError(f"No finite assay results for {column}")
    return float(grouped.groupby("selection").mean().mean())


def protein_cluster_bootstrap(table, candidate="The_Mountain", reference_method="PoET", repeats=5000, seed=0):
    """Paired bootstrap of whole proteins with the hierarchy recomputed each time."""
    if repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    columns = [candidate, reference_method]
    grouped = table.dropna(subset=columns).groupby(["uniprot", "selection"])[columns].mean()
    proteins = grouped.index.get_level_values("uniprot").unique().tolist()
    if not proteins:
        raise ValueError("No paired finite results for bootstrap")
    selections = grouped.index.get_level_values("selection").unique().tolist()
    # Multiplicity weights preserve duplicate proteins, while vectorization avoids
    # constructing thousands of temporary DataFrames.
    values = np.full((len(proteins), len(selections), 2), np.nan)
    for pi, protein in enumerate(proteins):
        for si, selection in enumerate(selections):
            if (protein, selection) in grouped.index:
                values[pi, si] = grouped.loc[(protein, selection), columns].to_numpy()
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats)
    for repeat in range(repeats):
        sampled = rng.integers(len(proteins), size=len(proteins))
        available = np.isfinite(values[sampled])
        sums = np.where(available, values[sampled], 0).sum(axis=0)
        counts = available.sum(axis=0)
        means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        aggregate = np.nanmean(means, axis=0)
        differences[repeat] = aggregate[0] - aggregate[1]
    return {
        "candidate": candidate, "reference": reference_method,
        "repeats": repeats, "seed": seed,
        "point_difference": official_aggregate(table, candidate) - official_aggregate(table, reference_method),
        "bootstrap_mean": float(differences.mean()),
        "ci_2.5": float(np.percentile(differences, 2.5)),
        "ci_97.5": float(np.percentile(differences, 97.5)),
        "fraction_positive": float(np.mean(differences > 0)),
    }, differences


def evaluate_assay(predictions, labels, *, prediction_column="The_Mountain"):
    """Join labels by stable IDs and verify sequence identity, never by position."""
    required = {"variant_id", "mutated_sequence"}
    if prediction_column not in predictions or "DMS_score" not in labels:
        raise ValueError("Prediction score column and experimental DMS_score labels are required")
    for frame in (predictions, labels):
        if not required.issubset(frame):
            raise ValueError(f"Missing identity columns: {required - set(frame)}")
        if frame.variant_id.isna().any() or frame.variant_id.duplicated().any():
            raise ValueError("Variant IDs must be unique and nonempty")
    if set(predictions.variant_id) != set(labels.variant_id):
        raise ValueError("Prediction and label variant IDs differ")
    merged = predictions.merge(labels[["variant_id", "mutated_sequence", "DMS_score"]], on="variant_id", validate="one_to_one", suffixes=("", "_label"))
    if not merged.mutated_sequence.equals(merged.mutated_sequence_label):
        raise ValueError("Sequence mismatch between prediction and label identities")
    truth = pd.to_numeric(merged.DMS_score, errors="raise").to_numpy(float)
    prediction = pd.to_numeric(merged[prediction_column], errors="raise").to_numpy(float)
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Evaluation requires finite labels and predictions")
    if len(truth) < 2 or np.unique(truth).size < 2 or np.unique(prediction).size < 2:
        raise ValueError("Spearman requires at least two variants and nonconstant labels/predictions")
    result = float(spearmanr(truth, prediction).statistic)
    if not np.isfinite(result):
        raise ValueError("Undefined Spearman correlation")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--column", default="mountain_score")
    args = parser.parse_args(argv)
    score = evaluate_assay(pd.read_csv(args.predictions, dtype={"variant_id": str, "mutated_sequence": str}), pd.read_csv(args.labels, dtype={"variant_id": str, "mutated_sequence": str}), prediction_column=args.column)
    print(json.dumps({"metric": "Spearman", "column": args.column, "score": score}, allow_nan=False))


if __name__ == "__main__":
    main()

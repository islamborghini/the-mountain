"""The Mountain's frozen, label-free rank ensemble.

Scores are relative to the variants supplied for one protein. Higher component
scores must mean greater predicted fitness. No experimental labels are used.
"""

import numpy as np
from scipy.stats import rankdata

PUBLISHED_SIGNALS = [
    "Unirep", "Wavenet", "RITA_s", "RITA_m", "RITA_l", "RITA_xl",
    "Progen2_small", "Progen2_medium", "Progen2_base", "Progen2_large",
    "Progen2_xlarge", "ProtGPT2", "Tranception_S_no_retrieval",
    "Tranception_M_no_retrieval", "Tranception_L_no_retrieval",
    "Tranception_S", "Tranception_M", "Tranception_L", "TranceptEVE_S",
    "TranceptEVE_M", "TranceptEVE_L", "HMM", "Provean",
]
SIGNALS = PUBLISHED_SIGNALS + ["ESM2_650M"]
SEQUENCE_SIGNALS = ["RITA_s", "RITA_m", "RITA_l", "RITA_xl", "ProtGPT2", "ESM2_650M"]
MODEL_VERSION = "mountain-average-ties-v1"


def rank_01(values):
    """Average tied ranks; identical inputs stay identical after row shuffles."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("rank_01 requires a nonempty finite one-dimensional array")
    return (rankdata(values, method="average") - 0.5) / values.size


def prepare_ranks(signals, signal_names):
    """Validate one protein's scores and omit wholly unavailable columns."""
    names = list(signal_names)
    if len(set(names)) != len(names):
        raise ValueError("signal_names must be unique")
    unknown = [name for name in names if name not in SIGNALS]
    if unknown:
        raise ValueError(f"Unsupported signals: {unknown}; PoET is a comparison baseline, not an input")
    try:
        matrix = np.asarray(signals, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("signals must contain numeric scores") from exc
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("signals must be a two-dimensional matrix with one column per signal_name")
    if not matrix.shape[0]:
        raise ValueError("At least one variant is required")
    present = ~np.all(np.isnan(matrix), axis=0)
    if not present.any():
        raise ValueError("No available signals: supply at least one complete score column")
    matrix = matrix[:, present]
    names = np.asarray(names, dtype=str)[present]
    bad = ~np.isfinite(matrix).all(axis=0)
    if bad.any():
        raise ValueError(f"Signals must be finite for every variant; incomplete or infinite columns: {names[bad].tolist()}")
    ranks = np.column_stack([rank_01(column) for column in matrix.T])
    is_esm2 = names == "ESM2_650M"
    return ranks, is_esm2, ~is_esm2


def score_variants(signals, signal_names, method="mountain"):
    """Return a finite score per variant using the fixed structural rule.

    All-NaN columns may represent unavailable models. Partially missing columns
    are rejected. A subset of the 24 signals is a different inference profile
    and does not inherit the full ensemble's benchmark result.
    """
    aliases = {"clean_full": "mountain", "clean_no_agreement": "no_agreement", "clean_no_split": "no_split"}
    method = aliases.get(method, method)
    if method not in {"mountain", "equal_weight", "no_agreement", "no_split"}:
        raise ValueError(f"Unknown ensemble method: {method}")
    ranks, is_esm2, is_published = prepare_ranks(signals, signal_names)
    if method == "equal_weight":
        return ranks.mean(axis=1)
    agreement = 1 / (1 + np.std(ranks, axis=1) ** 2)
    if method == "no_split":
        return np.exp(np.log(np.clip(ranks, 1e-9, 1)).mean(axis=1)) * agreement
    crowd = np.median(ranks[:, is_published], axis=1) if is_published.any() else None
    esm2 = ranks[:, is_esm2].mean(axis=1) if is_esm2.any() else None
    if crowd is not None and esm2 is not None:
        base = np.sqrt(np.clip(crowd, 1e-9, 1) * np.clip(esm2, 1e-9, 1))
    else:
        base = crowd if crowd is not None else esm2
    return base if method == "no_agreement" else base * agreement

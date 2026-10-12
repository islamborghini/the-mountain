"""Generate fixed protein-model signals without labels or research services.

RITA and ProtGPT2 follow ProteinGym's forward/reverse chunk scoring. ESM2
uses the original Mountain's unmasked observed-token confidence, not PLL.
"""

from pathlib import Path
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import tempfile

import numpy as np


CHECKPOINTS = {
    "RITA_s": ("lightonai/RITA_s", "818eb05afad820ba5fd6485b224559932eccd01f"),
    "RITA_m": ("lightonai/RITA_m", "d819157a3a96d278500232b2cbb2a02d9646bcf6"),
    "RITA_l": ("lightonai/RITA_l", "b38b5daa35b416f710216305c7258e3138503b62"),
    "RITA_xl": ("lightonai/RITA_xl", "dc89dfa8783ad120ef9a1751b8cea7ddc1511902"),
    "ProtGPT2": ("nferruz/ProtGPT2", "ff981fc96771d6f6d3b6453f94772d3cb6c7b90b"),
    "ESM2_650M": ("facebook/esm2_t33_650M_UR50D", "08e4846e537177426273712802403f7ba8261b6c"),
}
DEFAULT_SIGNALS = tuple(CHECKPOINTS)
SCORING_VERSION = 1
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def ordered_sequences_sha256(sequences):
    """Hash exact sequence content and order; equal lengths are insufficient."""
    payload = json.dumps(list(sequences), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dependencies():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Sequence inference requires optional dependencies: "
            "python -m pip install '.[inference]'"
        ) from exc
    return torch, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


def _device(torch, requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda[:index], or mps")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def _feature_config(name, length_policy):
    repository, revision = CHECKPOINTS[name]
    return {
        "name": name, "repository": repository, "revision": revision,
        "scoring_version": SCORING_VERSION,
        "scoring_policy": (
            "unmasked_mean_observed_residue_logp" if name == "ESM2_650M" else
            "sum_forward_reverse_chunk_mean_logp" if name.startswith("RITA") else
            "mean_forward_reverse_chunk_mean_logp"
        ),
        "length_policy": length_policy if name == "ESM2_650M" else "chunk",
        "max_residues": 1022 if name == "ESM2_650M" else 1023,
        "precision": "float32",
    }


def _transformers_version():
    try:
        return version("transformers")
    except PackageNotFoundError:
        return "unavailable"


def _load_checkpoint(name, device, cache_dir):
    torch, causal, masked, tokenizer_class = _dependencies()
    repository, revision = CHECKPOINTS[name]
    kwargs = {"revision": revision, "cache_dir": cache_dir}
    tokenizer = tokenizer_class.from_pretrained(repository, **kwargs)
    model_class = masked if name == "ESM2_650M" else causal
    if name.startswith("RITA"):
        # RITA's architecture is custom; both weights and executed source are pinned.
        kwargs.update(trust_remote_code=True, code_revision=revision)
    model = model_class.from_pretrained(repository, torch_dtype=torch.float32, **kwargs)
    return tokenizer, model.to(device).eval()


def _esm2_batches(sequences, indices, tokenizer, model, device, length_policy, batch_tokens):
    import torch

    order = sorted(indices, key=lambda i: min(len(sequences[i]), 1022))
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order):
            longest = min(len(sequences[order[stop]]), 1022) + 2
            if (stop - start + 1) * longest > batch_tokens:
                break
            stop += 1
        batch_indices = order[start:stop]
        start = stop
        batch = [sequences[i][:1022] if length_policy == "truncate" else sequences[i]
                 for i in batch_indices]
        encoded = tokenizer(batch, return_tensors="pt", padding=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits.float()
            token_ids = encoded["input_ids"]
            observed = logits.log_softmax(-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
            special = torch.zeros_like(token_ids, dtype=torch.bool)
            for token_id in tokenizer.all_special_ids:
                special |= token_ids == token_id
            valid = encoded["attention_mask"].bool() & ~special
            if (valid.sum(-1) == 0).any():
                raise ValueError("ESM2 tokenizer produced a sequence without residue tokens")
            values = ((observed * valid).sum(-1) / valid.sum(-1)).cpu().numpy()
        yield batch_indices, values


def _causal_score(sequence, name, tokenizer, model, device):
    import torch

    losses = []
    # range avoids ProteinGym's empty final chunk at an exact context multiple.
    for start in range(0, len(sequence), 1023):
        chunk = sequence[start:start + 1023]
        for direction in (chunk, chunk[::-1]):
            token_ids = tokenizer.encode(direction)
            if len(token_ids) < 2:
                raise ValueError(
                    f"{name}: a {len(chunk)}-residue chunk has fewer than two tokens; "
                    "next-token likelihood is undefined"
                )
            if len(token_ids) - 1 > 1024:
                raise ValueError(f"{name}: encoded chunk exceeds model context")
            ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                logits = model(input_ids=ids[:, :-1]).logits.float()
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
                )
            losses.append(-loss.item())
    return float(sum(losses) if name.startswith("RITA") else np.mean(losses))


def _save_cache(path, scores, signature):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem, suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, scores=scores, signature=np.asarray(signature))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def generate_signals(sequences, signal_names=None, device="auto", cache_dir=None,
                     length_policy="error", batch_tokens=1500):
    """Return (score matrix, names, JSON metadata) for canonical protein sequences.

    All models are loaded one at a time. Cache entries bind exact ordered input,
    checkpoint revision, device, precision and scoring policy, and save each
    completed batch so a cancelled run can resume. Model downloads use the
    normal Hugging Face cache; ``cache_dir`` stores computed scores.

    ESM2 rejects sequences beyond 1022 residues unless ``length_policy='truncate'``
    is explicit. Causal models score all residues using 1023-residue chunks.
    """
    sequences = list(sequences)
    names = list(DEFAULT_SIGNALS if signal_names is None else signal_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Choose at least one model, without duplicate model names")
    unknown = set(names) - CHECKPOINTS.keys()
    if unknown:
        raise ValueError(f"Unsupported sequence models: {sorted(unknown)}; available: {list(CHECKPOINTS)}")
    if length_policy not in {"error", "truncate"}:
        raise ValueError("length_policy must be error or truncate")
    if isinstance(batch_tokens, bool) or not isinstance(batch_tokens, int) or batch_tokens < 3:
        raise ValueError("batch_tokens must be an integer of at least 3")
    if not sequences:
        raise ValueError("Provide at least one protein sequence")
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, str) or not sequence or set(sequence) - AMINO_ACIDS:
            raise ValueError(f"Sequence {index} must contain only uppercase canonical amino acids")
    if "ESM2_650M" in names and length_policy == "error":
        too_long = [i for i, sequence in enumerate(sequences) if len(sequence) > 1022]
        if too_long:
            raise ValueError(
                f"ESM2 accepts at most 1022 residues; sequence {too_long[0]} has "
                f"{len(sequences[too_long[0]])}. Set length_policy='truncate' explicitly "
                "to score only the first 1022 residues."
            )
    torch, _, _, _ = _dependencies()
    selected_device = _device(torch, device)
    transformers_version = _transformers_version()
    digest = ordered_sequences_sha256(sequences)
    features = [_feature_config(name, length_policy) for name in names]
    output = np.empty((len(sequences), len(names)), dtype=np.float64)
    for column, (name, feature) in enumerate(zip(names, features)):
        provenance = {
            "ordered_sequences_sha256": digest, "feature": feature,
            "device": str(selected_device), "torch_version": torch.__version__,
            "transformers_version": transformers_version,
        }
        signature = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        cache_path = None if cache_dir is None else Path(cache_dir) / (
            hashlib.sha256(signature.encode()).hexdigest() + ".npz"
        )
        scores = np.full(len(sequences), np.nan, dtype=np.float64)
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as stored:
                if str(stored["signature"]) != signature:
                    raise ValueError(f"Score cache provenance mismatch: {cache_path}")
                scores = np.asarray(stored["scores"], dtype=np.float64)
                if scores.shape != (len(sequences),) or np.isinf(scores).any():
                    raise ValueError(f"Invalid score cache: {cache_path}")
        missing = np.flatnonzero(np.isnan(scores)).tolist()
        if missing:
            tokenizer, model = _load_checkpoint(name, selected_device, None)
            try:
                if name == "ESM2_650M":
                    batches = _esm2_batches(sequences, missing, tokenizer, model,
                                            selected_device, length_policy, batch_tokens)
                else:
                    batches = (([i], [_causal_score(sequences[i], name, tokenizer, model,
                                                    selected_device)]) for i in missing)
                for indices, values in batches:
                    if not np.isfinite(values).all():
                        raise RuntimeError(f"{name} produced nonfinite scores")
                    scores[indices] = values
                    if cache_path is not None:
                        _save_cache(cache_path, scores, signature)
            finally:
                del model, tokenizer
                gc.collect()
                if selected_device.type == "cuda":
                    torch.cuda.empty_cache()
                elif selected_device.type == "mps":
                    torch.mps.empty_cache()
        output[:, column] = scores
    metadata = {
        "ordered_sequences_sha256": digest, "n_sequences": len(sequences),
        "device": str(selected_device), "features": features,
        "torch_version": torch.__version__, "original_method": False,
        "transformers_version": transformers_version,
        "profile": "sequence_6" if names == list(DEFAULT_SIGNALS) else "partial_sequence_ensemble",
        "truncated_sequences": sum(len(s) > 1022 for s in sequences)
        if "ESM2_650M" in names and length_policy == "truncate" else 0,
    }
    return output, names, metadata

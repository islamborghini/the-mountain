import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import component_scoring as scoring


class CausalTokenizer:
    def encode(self, sequence):
        return [1 if residue == "A" else 2 for residue in sequence] + [3]


class CausalModel:
    def __call__(self, input_ids):
        # Nonuniform next-token probabilities reveal wrong shifts/direction.
        logits = torch.zeros((*input_ids.shape, 4), device=input_ids.device)
        logits[..., 1] = input_ids.float()
        return SimpleNamespace(logits=logits)


def test_causal_forward_reverse_and_model_specific_reductions():
    tokenizer, model = CausalTokenizer(), CausalModel()
    directions = ["ACAA", "AACA"]
    expected = []
    for sequence in directions:
        ids = torch.tensor([tokenizer.encode(sequence)])
        logits = model(ids[:, :-1]).logits
        expected.append(-torch.nn.functional.cross_entropy(
            logits.reshape(-1, 4), ids[:, 1:].reshape(-1)
        ).item())
    rita = scoring._causal_score("ACAA", "RITA_s", tokenizer, model, "cpu")
    gpt = scoring._causal_score("ACAA", "ProtGPT2", tokenizer, model, "cpu")
    assert rita == pytest.approx(sum(expected))
    assert gpt == pytest.approx(np.mean(expected))


def test_causal_exact_chunk_multiple_has_no_empty_chunk():
    class CountingTokenizer(CausalTokenizer):
        chunks = []

        def encode(self, sequence):
            self.chunks.append(sequence)
            assert sequence
            return super().encode(sequence)

    tokenizer = CountingTokenizer()
    result = scoring._causal_score("A" * 2046, "RITA_s", tokenizer, CausalModel(), "cpu")
    assert np.isfinite(result)
    assert len(tokenizer.chunks) == 4
    assert all(len(chunk) == 1023 for chunk in tokenizer.chunks)


def test_single_token_causal_input_fails_clearly():
    tokenizer = SimpleNamespace(encode=lambda sequence: [1])
    with pytest.raises(ValueError, match="next-token likelihood is undefined"):
        scoring._causal_score("A", "ProtGPT2", tokenizer, CausalModel(), "cpu")


class EsmTokenizer:
    all_special_ids = [0, 3]

    def __call__(self, sequences, return_tensors, padding):
        encoded = [[0] + [1 if residue == "A" else 2 for residue in s] + [3]
                   for s in sequences]
        width = max(map(len, encoded))
        masks = [[1] * len(s) + [0] * (width - len(s)) for s in encoded]
        encoded = [s + [0] * (width - len(s)) for s in encoded]
        return {"input_ids": torch.tensor(encoded), "attention_mask": torch.tensor(masks)}


class EsmModel:
    def __call__(self, input_ids, attention_mask):
        logits = torch.zeros((*input_ids.shape, 4), device=input_ids.device)
        logits[..., 1] = 2
        return SimpleNamespace(logits=logits)


def test_esm_observed_tokens_excludes_specials_and_padding_and_keeps_order():
    sequences = ["AAA", "C", "AC"]
    results = np.full(3, np.nan)
    for indices, values in scoring._esm2_batches(
        sequences, [0, 1, 2], EsmTokenizer(), EsmModel(), "cpu", "error", 12
    ):
        results[indices] = values
    residue_logp = torch.tensor([0., 2., 0., 0.]).log_softmax(-1).numpy()
    assert results == pytest.approx([
        residue_logp[1], residue_logp[2], (residue_logp[1] + residue_logp[2]) / 2
    ])


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(scoring, "_dependencies", lambda: (torch, None, None, None))
    loaded = []

    def load(name, device, cache_dir):
        loaded.append(name)
        if name == "ESM2_650M":
            return EsmTokenizer(), EsmModel()
        return CausalTokenizer(), CausalModel()

    monkeypatch.setattr(scoring, "_load_checkpoint", load)
    return loaded


def test_native_cache_binds_sequence_order_and_settings(tmp_path, fake_runtime):
    signals, names, metadata = scoring.generate_signals(
        ["AAA", "C"], ["RITA_s", "ESM2_650M"], "cpu", tmp_path
    )
    assert signals.shape == (2, 2)
    assert np.isfinite(signals).all()
    assert names == ["RITA_s", "ESM2_650M"]
    json.dumps(metadata)
    assert not metadata["original_method"]
    assert fake_runtime == names
    replay, _, _ = scoring.generate_signals(["AAA", "C"], names, "cpu", tmp_path)
    assert np.array_equal(replay, signals)
    assert fake_runtime == names  # No checkpoint is loaded for complete cache hits.
    shuffled, _, _ = scoring.generate_signals(["C", "AAA"], names, "cpu", tmp_path)
    assert shuffled[::-1] == pytest.approx(signals)
    assert fake_runtime == names * 2
    assert len(list(tmp_path.glob("*.npz"))) == 4


def test_partial_cache_resumes_after_interruption(tmp_path, fake_runtime, monkeypatch):
    original = scoring._causal_score
    calls = []

    def interrupted(sequence, *args):
        calls.append(sequence)
        if len(calls) == 2:
            raise RuntimeError("cancelled")
        return original(sequence, *args)

    monkeypatch.setattr(scoring, "_causal_score", interrupted)
    with pytest.raises(RuntimeError, match="cancelled"):
        scoring.generate_signals(["AAA", "C"], ["RITA_s"], "cpu", tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    calls.clear()

    def resumed(sequence, *args):
        calls.append(sequence)
        return original(sequence, *args)

    monkeypatch.setattr(scoring, "_causal_score", resumed)
    result, _, _ = scoring.generate_signals(["AAA", "C"], ["RITA_s"], "cpu", tmp_path)
    assert np.isfinite(result).all()
    assert calls == ["C"]


@pytest.mark.parametrize("sequences", [[], [""], ["ac"], ["A X"], ["AX"], [None]])
def test_invalid_sequences_rejected_before_model_download(sequences):
    with pytest.raises(ValueError):
        scoring.generate_signals(sequences)


def test_esm2_length_policy_explicit(fake_runtime):
    with pytest.raises(ValueError, match="1022 residues"):
        scoring.generate_signals(["A" * 1023], ["ESM2_650M"])
    values, _, metadata = scoring.generate_signals(
        ["A" * 1023], ["ESM2_650M"], "cpu", length_policy="truncate"
    )
    assert np.isfinite(values).all()
    assert metadata["truncated_sequences"] == 1
    assert metadata["features"][0]["length_policy"] == "truncate"


def test_unsupported_models_rejected_before_loading():
    with pytest.raises(ValueError, match="Unsupported sequence models"):
        scoring.generate_signals(["ACD"], ["HMM"])
    with pytest.raises(ValueError, match="duplicate"):
        scoring.generate_signals(["ACD"], ["RITA_s", "RITA_s"])


def test_sequence_digest_contains_order_and_content():
    assert scoring.ordered_sequences_sha256(["AA", "CC"]) != scoring.ordered_sequences_sha256(["CC", "AA"])
    assert scoring.ordered_sequences_sha256(["AA", "CC"]) != scoring.ordered_sequences_sha256(["AC", "CA"])


def test_all_checkpoint_revisions_are_immutable():
    assert len(scoring.DEFAULT_SIGNALS) == 6
    for repository, revision in scoring.CHECKPOINTS.values():
        assert "/" in repository
        assert len(revision) == 40
        int(revision, 16)


def test_rita_loader_pins_weights_and_executed_code(monkeypatch):
    calls = []

    class Loader:
        @classmethod
        def from_pretrained(cls, repository, **kwargs):
            calls.append((repository, kwargs))
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(scoring, "_dependencies", lambda: (torch, Loader, Loader, Loader))
    scoring._load_checkpoint("RITA_s", "cpu", None)
    repository, revision = scoring.CHECKPOINTS["RITA_s"]
    assert all(repo == repository and kwargs["revision"] == revision for repo, kwargs in calls)
    assert calls[-1][1]["code_revision"] == revision
    assert calls[-1][1]["trust_remote_code"] is True


def test_esm_truncation_drops_tail_but_preserves_provenance(fake_runtime):
    values, _, metadata = scoring.generate_signals(
        ["A" * 1022, "A" * 1022 + "C"], ["ESM2_650M"], "cpu", length_policy="truncate"
    )
    assert values[0, 0] == values[1, 0]
    assert metadata["ordered_sequences_sha256"] == scoring.ordered_sequences_sha256(
        ["A" * 1022, "A" * 1022 + "C"]
    )

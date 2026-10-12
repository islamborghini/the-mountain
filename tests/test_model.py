import numpy as np
import pytest

from model import rank_01, score_variants


def test_ties_and_row_permutation_invariance():
    matrix = np.array([[1, 4, 2], [1, 4, 2], [3, 1, 4], [4, 2, 4], [2, 3, 1]])
    names = ["RITA_s", "ProtGPT2", "ESM2_650M"]
    for method in ["mountain", "equal_weight", "no_agreement", "no_split"]:
        expected = score_variants(matrix, names, method)
        permutation = np.array([4, 0, 3, 1, 2])
        shuffled = score_variants(matrix[permutation], names, method)
        np.testing.assert_allclose(shuffled[np.argsort(permutation)], expected, atol=0, rtol=0)
        assert expected[0] == expected[1]
    np.testing.assert_array_equal(rank_01([2, 2, 2]), [0.5, 0.5, 0.5])


def test_frozen_structural_rule():
    matrix = np.array([[1, 3, 2], [3, 1, 3], [2, 2, 1]])
    ranks = np.array([[1 / 6, 5 / 6, 1 / 2], [5 / 6, 1 / 6, 5 / 6], [1 / 2, 1 / 2, 1 / 6]])
    expected = np.sqrt(np.median(ranks[:, :2], axis=1) * ranks[:, 2]) / (1 + np.std(ranks, axis=1) ** 2)
    np.testing.assert_allclose(score_variants(matrix, ["RITA_s", "ProtGPT2", "ESM2_650M"]), expected)


def test_unavailable_columns_and_single_variant():
    values = score_variants([[1, np.nan], [3, np.nan]], ["RITA_s", "ESM2_650M"])
    np.testing.assert_array_equal(values, [0.25, 0.75])
    np.testing.assert_array_equal(score_variants([[4, 3]], ["RITA_s", "ESM2_650M"]), [0.5])


@pytest.mark.parametrize("matrix,names,match", [
    ([[1], [np.nan]], ["RITA_s"], "incomplete"),
    ([[1], [np.inf]], ["RITA_s"], "infinite"),
    ([[np.nan]], ["RITA_s"], "No available"),
    (np.empty((0, 1)), ["RITA_s"], "At least one variant"),
    ([[1]], [], "one column"),
    ([[1]], ["PoET"], "Unsupported"),
    ([[1]], ["new_model"], "Unsupported"),
    ([[1, 2]], ["RITA_s", "RITA_s"], "unique"),
    ([["oops"]], ["RITA_s"], "numeric"),
])
def test_invalid_inputs(matrix, names, match):
    with pytest.raises(ValueError, match=match):
        score_variants(matrix, names)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown ensemble"):
        score_variants([[1]], ["RITA_s"], "made_up")

import unittest
import io
import json
import tempfile
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from evaluate import evaluate_assay, official_aggregate, protein_cluster_bootstrap, main


class EvaluationTests(unittest.TestCase):
    def test_identity_join_and_sequence_check(self):
        pred = pd.DataFrame({'variant_id': ['b', 'a', 'c'], 'mutated_sequence': ['CC', 'AA', 'DD'], 'The_Mountain': [2., 1., 3.]})
        labels = pd.DataFrame({'variant_id': ['a', 'b', 'c'], 'mutated_sequence': ['AA', 'CC', 'DD'], 'DMS_score': [1., 2., 3.]})
        self.assertEqual(evaluate_assay(pred, labels), 1.)
        labels.loc[1, 'mutated_sequence'] = 'EE'
        with self.assertRaisesRegex(ValueError, 'Sequence mismatch'):
            evaluate_assay(pred, labels)

    def test_mismatched_or_duplicate_ids_fail(self):
        frame = pd.DataFrame({'variant_id': ['a', 'a'], 'mutated_sequence': ['AA', 'BB'], 'The_Mountain': [1., 2.], 'DMS_score': [1., 2.]})
        with self.assertRaisesRegex(ValueError, 'unique'):
            evaluate_assay(frame, frame)

    def test_cli_preserves_leading_zero_variant_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            predictions = Path(folder) / 'predictions.csv'
            labels = Path(folder) / 'labels.csv'
            predictions.write_text('variant_id,mutated_sequence,mountain_score\n01,AA,1\n1,CC,2\n')
            labels.write_text('variant_id,mutated_sequence,DMS_score\n1,CC,2\n01,AA,1\n')
            output = io.StringIO()
            with redirect_stdout(output):
                main(['--predictions', str(predictions), '--labels', str(labels)])
            self.assertAlmostEqual(json.loads(output.getvalue())["score"], 1.)

    def test_undefined_correlation_fails_clearly(self):
        frame = pd.DataFrame({"variant_id": ["01", "02"], "mutated_sequence": ["AA", "BB"], "The_Mountain": [1., 1.], "DMS_score": [1., 2.]})
        with self.assertRaisesRegex(ValueError, "nonconstant"):
            evaluate_assay(frame, frame)

    def test_hierarchy_does_not_weight_protein_by_assay_count(self):
        table = pd.DataFrame({'uniprot': ['a', 'a', 'b', 'c'], 'selection': ['X', 'X', 'X', 'Y'], 'score': [0., 0., 1., 1.]})
        self.assertEqual(official_aggregate(table, 'score'), .75)
        table['score'] = [.12345, .12345, .34567, .56789]
        expected = ((.123 + .346)/2 + .568)/2
        self.assertAlmostEqual(official_aggregate(table, 'score', round_assays=True), expected)

    def test_bootstrap_is_paired_clustered_and_deterministic(self):
        table = pd.DataFrame({'uniprot': ['a', 'a', 'b', 'c'], 'selection': ['X', 'X', 'X', 'Y'], 'The_Mountain': [.5, .7, .9, .8], 'PoET': [.3, .5, .7, .6]})
        first, draws = protein_cluster_bootstrap(table, repeats=100, seed=4)
        second, again = protein_cluster_bootstrap(table, repeats=100, seed=4)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(draws, again)
        np.testing.assert_allclose(draws, .2)


if __name__ == '__main__':
    unittest.main()

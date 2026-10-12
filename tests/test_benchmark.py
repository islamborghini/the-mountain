import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd

from benchmark import cache_esm2, load_reference, read_esm2, validated_members, run_benchmark, sequence_hash, SKIP_ESM2
from model import PUBLISHED_SIGNALS


class BenchmarkTests(unittest.TestCase):
    def test_sequence_bound_cache_rejects_same_length_reorder(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'assay.npy'
            cache_esm2(path, [1., 2.], ['AA', 'CC'], {'kind': 'test'})
            values, metadata = read_esm2(path, ['AA', 'CC'])
            np.testing.assert_array_equal(values, [1., 2.])
            self.assertEqual(metadata['provenance']['kind'], 'test')
            with self.assertRaisesRegex(ValueError, 'sequence mismatch'):
                read_esm2(path, ['CC', 'AA'])
            with path.open('wb') as stream:
                np.save(stream, [3., 4.])
            with self.assertRaisesRegex(ValueError, 'content hash mismatch'):
                read_esm2(path, ['AA', 'CC'])

    def test_raw_legacy_npy_cannot_silently_enter(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'assay.npy'
            np.save(path, [1., 2.])
            with self.assertRaisesRegex(ValueError, 'require sequence-hash'):
                read_esm2(path, ['AA', 'CC'])

    def test_archive_rejects_unsafe_names_duplicates_and_omissions(self):
        for names in [['../assay.csv'], ['one/assay.csv', 'two/assay.csv'], ['different.csv']]:
            stream = io.BytesIO()
            with ZipFile(stream, 'w') as archive:
                for name in names:
                    archive.writestr(name, 'sequence,score\nAA,1\n')
            stream.seek(0)
            with ZipFile(stream) as archive:
                with self.assertRaises(ValueError):
                    validated_members(archive, '.csv', ['assay'])

    def test_native_benchmark_regenerates_causal_features_without_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive_path = root / 'scores.zip'
            assays = ['example', next(iter(SKIP_ESM2))]
            specs = {}
            with ZipFile(archive_path, 'w') as archive:
                for assay in assays:
                    frame = pd.DataFrame({'mutated_sequence': ['AA', 'CC', 'DD'],
                                          'DMS_score': [3., 1., 2.],
                                          'DMS_score_bin': [1, 0, 1], 'PoET': [3., 1., 2.]})
                    for name in PUBLISHED_SIGNALS:
                        frame[name] = [3., 1., 2.]
                    raw = frame.to_csv(index=False).encode()
                    archive.writestr(assay + '.csv', raw)
                    specs[assay] = {'rows': 3, 'score_csv_sha256': hashlib.sha256(raw).hexdigest(),
                                    'ordered_sequences_sha256': sequence_hash(frame.mutated_sequence)}
            reference = pd.DataFrame({'DMS_id': assays, 'UniProt_ID': ['P1', 'P2'],
                                      'coarse_selection_type': ['Activity', 'Activity']})
            manifest = {'assays': specs, 'reference_sha256': 'mock', 'upstream_evaluator_commit': 'mock'}
            calls = []
            def fake_generate(sequences, signal_names, **kwargs):
                calls.append((list(sequences), list(signal_names)))
                values = np.tile(np.arange(1, len(sequences) + 1, dtype=float)[:, None], (1, len(signal_names)))
                return values, signal_names, {'mock_generated': True}
            with patch('benchmark.load_reference', return_value=(reference, manifest)), \
                 patch('benchmark.download_scores'), \
                 patch('component_scoring.generate_signals', side_effect=fake_generate):
                table, summary = run_benchmark(root, scores_zip=archive_path, profile='sequence_6',
                                                generate_components=True, bootstrap_repeats=10)
            self.assertEqual(calls[0], (['AA', 'CC', 'DD'] * 2, ['RITA_s', 'RITA_m', 'RITA_l', 'RITA_xl', 'ProtGPT2']))
            self.assertEqual(calls[1], (['AA', 'CC', 'DD'], ['ESM2_650M']))
            prediction = pd.read_csv(root / 'results' / 'predictions' / 'example.csv')
            self.assertNotIn('DMS_score', prediction)
            # Native mock scores rise by row; published scores peak at first row.
            self.assertTrue(np.all(np.diff(prediction.The_Mountain) > 0))
            saved = json.loads((root / 'results' / 'run_manifest.json').read_text())
            self.assertEqual(saved['causal_feature_source'], 'native_checkpoint_scoring')
            self.assertEqual(saved['native_causal_feature_metadata'], {'mock_generated': True})

    def test_native_original_profile_rejected_before_io(self):
        with self.assertRaisesRegex(ValueError, 'only the native sequence_6'):
            run_benchmark('/unused', generate_components=True)

    def test_frozen_reference_is_complete(self):
        reference, manifest = load_reference()
        self.assertEqual(len(reference), 66)
        self.assertEqual(set(reference.DMS_id), set(manifest['assays']))
        self.assertTrue(all(spec['rows'] > 0 for spec in manifest['assays'].values()))


if __name__ == '__main__':
    unittest.main()

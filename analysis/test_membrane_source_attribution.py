import unittest

import numpy as np

from analysis.analyze_membrane_sources import (
    ProportionalSourceLedger,
    _mirrored_power_norm,
    _log_heatmap_data,
    _source_matrix,
    plot_results,
    run_source_analysis,
)


class ProportionalSourceLedgerTest(unittest.TestCase):
    def test_user_example_tracks_first_input_through_two_resets(self):
        ledger = ProportionalSourceLedger(decay=1.0)
        ledger.charge(1.2)
        ledger.redistribute(0.2)
        self.assertAlmostEqual(ledger.sources[0], 0.2)

        ledger.charge(0.9)
        self.assertAlmostEqual(ledger.total, 1.1)
        ledger.redistribute(0.1)
        self.assertAlmostEqual(ledger.sources[0], 0.1 * 2.0 / 11.0)
        self.assertAlmostEqual(ledger.sources[1], 0.1 * 9.0 / 11.0)

        ledger.decay = 1.0 / 6.0
        ledger.charge(0.0)
        self.assertAlmostEqual(ledger.sources[0], 0.1 * 2.0 / 11.0 / 6.0)

    def test_sources_always_sum_to_observed_residual(self):
        ledger = ProportionalSourceLedger(decay=0.5)
        ledger.charge(1.2)
        ledger.redistribute(0.2)
        ledger.charge(0.9)
        ledger.redistribute(0.0)
        self.assertAlmostEqual(ledger.total, 0.0)
        self.assertTrue(all(value == 0.0 for value in ledger.sources))

    def test_nonzero_target_cannot_come_from_zero_sources(self):
        ledger = ProportionalSourceLedger(decay=0.5)
        ledger.charge(0.0)
        with self.assertRaisesRegex(ValueError, 'zero source total'):
            ledger.redistribute(0.1)

    def test_standalone_dynamics_reconstruct_every_decision(self):
        rows, source_rows = run_source_analysis([1.2, 0.9, 0.0, 0.0])
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(source_rows), 10)
        self.assertTrue(all(row['lif_reconstruction_error'] < 1e-12 for row in rows))
        self.assertTrue(all(row['lslif_reconstruction_error'] < 1e-12 for row in rows))

    def test_heatmap_gamma_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, 'heatmap_gamma must be positive'):
            plot_results([], [], 'unused.png', heatmap_gamma=0.0)

    def test_source_matrix_can_plot_absolute_contributions(self):
        source_rows = [{
            'source_step': 0,
            'decision_step': 1,
            'signed_percent': -25.0,
        }]
        matrix = _source_matrix(source_rows, 'signed_percent', 2, absolute=True)
        self.assertEqual(matrix[0, 1], 25.0)

    def test_reversed_palette_uses_a_mirrored_power_scale(self):
        norm = _mirrored_power_norm(vmax=100.0, gamma=0.35)
        self.assertLess(float(norm(20.0)), 0.2)
        self.assertGreater(float(norm(80.0)), 0.4)
        for value in (0.0, 20.0, 80.0, 100.0):
            self.assertAlmostEqual(float(norm.inverse(norm(value))), value)

    def test_log_heatmap_gives_equal_space_to_each_decade(self):
        lif = np.asarray([[100.0, 0.1], [np.nan, 0.0]])
        lslif = np.asarray([[10.0, 0.01], [1.0, 0.001]])
        lif_plot, lslif_plot, norm = _log_heatmap_data(lif, lslif, decades=5)
        self.assertAlmostEqual(float(norm(0.1) - norm(0.01)), 0.2)
        self.assertAlmostEqual(float(norm(0.01) - norm(0.001)), 0.2)
        self.assertEqual(lif_plot[1, 1], norm.vmin)
        self.assertTrue(np.isnan(lif_plot[1, 0]))
        self.assertEqual(lslif_plot[1, 1], 0.001)


if __name__ == '__main__':
    unittest.main()

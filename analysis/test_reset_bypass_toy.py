import importlib.util
import unittest

from analysis.reset_bypass_toy import ToyConfig, simulate, summarize


class ResetBypassToyTest(unittest.TestCase):
    def test_branch_is_unchanged_by_forced_reset(self):
        config = ToyConfig(steps=4, pulse_steps=4, pulse_amplitude=1.4, reset_steps=(2,))
        reset_row = simulate(config)[2]
        self.assertAlmostEqual(reset_row['absolute_reset_loss'], config.threshold)
        self.assertGreater(reset_row['history_branch'], 0.0)
        self.assertAlmostEqual(
            reset_row['ls_effective_post'] - reset_row['lif_effective_post'],
            reset_row['history_branch'],
        )

    def test_ls_improves_relative_retention_without_hiding_main_loss(self):
        rows = simulate(ToyConfig(steps=4, pulse_steps=4, pulse_amplitude=1.4, reset_steps=(2,)))
        metrics = summarize(rows)
        self.assertGreater(metrics['retention_ratio_gain'], 0.0)
        self.assertAlmostEqual(metrics['mean_absolute_main_reset_loss'], 1.0)

    def test_rejects_reset_when_membrane_could_not_have_spiked(self):
        with self.assertRaisesRegex(ValueError, 'not spike-valid'):
            simulate(ToyConfig(steps=2, pulse_amplitude=0.1, reset_steps=(0,)))


@unittest.skipUnless(
    importlib.util.find_spec('torch') and importlib.util.find_spec('spikingjelly'),
    'production-neuron comparison requires torch and spikingjelly',
)
class RealNeuronComparisonTest(unittest.TestCase):
    def test_executes_repository_neuron_classes(self):
        from analysis.reset_bypass_real_neurons import run_actual_neurons, summarize

        rows = run_actual_neurons(steps=8, pulse_steps=8)
        metrics = summarize(rows)
        self.assertEqual(metrics['implementation']['lif'], 'modules.neuron.VanillaLIFNeuron')
        self.assertEqual(metrics['implementation']['lslif'], 'modules.neuron.LSLIFNeuron')
        self.assertGreater(metrics['lif_spike_count'], 0)
        self.assertGreater(metrics['lslif_spike_count'], 0)
        self.assertGreater(metrics['mean_lslif_preserved_branch_at_reset'], 0.0)


if __name__ == '__main__':
    unittest.main()

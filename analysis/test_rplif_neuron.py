import unittest

import torch

from modules.neuron import LSRPLIFNeuron


class LSRPLIFHistoryStartTest(unittest.TestCase):
    def test_history_is_accumulated_but_not_fused_before_start_step(self):
        neuron = LSRPLIFNeuron(
            tau=2.0,
            v_threshold=1.5,
            history_weight=1.0,
            history_power=1.0,
            lsrplif_history_start_step=3,
        )

        first_spike = neuron(torch.tensor([0.8]))
        self.assertEqual(first_spike.item(), 0.0)
        self.assertAlmostEqual(neuron.n.item(), 0.8)

        second_spike = neuron(torch.tensor([0.0]))
        self.assertEqual(second_spike.item(), 0.0)
        self.assertAlmostEqual(neuron.n.item(), 0.4)

        third_spike = neuron(torch.tensor([1.0]))
        self.assertEqual(third_spike.item(), 1.0)

    def test_default_still_fuses_history_from_first_step(self):
        neuron = LSRPLIFNeuron(
            tau=2.0,
            v_threshold=1.5,
            history_weight=1.0,
            history_power=1.0,
        )
        self.assertEqual(neuron(torch.tensor([0.8])).item(), 1.0)

    def test_start_step_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, 'must be at least 1'):
            LSRPLIFNeuron(lsrplif_history_start_step=0)


if __name__ == '__main__':
    unittest.main()

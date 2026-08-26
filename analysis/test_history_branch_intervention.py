import unittest

import torch

from analysis.history_branch_intervention_eval import parse_condition
from modules.neuron import LSLIFNeuron


class HistoryBranchInterventionTest(unittest.TestCase):
    def test_zero_removes_history_contribution(self):
        neuron = LSLIFNeuron(tau=2.0, history_weight=1.0, history_power=0.0)
        neuron.set_history_intervention('zero')
        spike = neuron(torch.tensor([[0.6]]))
        self.assertEqual(float(spike.item()), 0.0)
        self.assertAlmostEqual(float(neuron.n.item()), 0.6, places=6)
        self.assertAlmostEqual(float(neuron.v.item()), 0.6, places=6)

    def test_shuffle_rolls_batch_history(self):
        neuron = LSLIFNeuron()
        term = torch.tensor([[1.0], [2.0], [3.0]])
        neuron.set_history_intervention('shuffle')
        self.assertTrue(torch.equal(neuron._intervene_history_term(term), torch.tensor([[3.0], [1.0], [2.0]])))

    def test_time_shift_buffer_is_cleared_by_reset(self):
        neuron = LSLIFNeuron()
        neuron.set_history_intervention('time_shift', shift=2)
        self.assertEqual(float(neuron._intervene_history_term(torch.tensor([1.0]))), 0.0)
        self.assertEqual(float(neuron._intervene_history_term(torch.tensor([2.0]))), 0.0)
        self.assertEqual(float(neuron._intervene_history_term(torch.tensor([3.0]))), 1.0)
        neuron.reset()
        self.assertEqual(float(neuron._intervene_history_term(torch.tensor([4.0]))), 0.0)

    def test_condition_parser(self):
        self.assertEqual(parse_condition('time-shift-4'), ('time_shift_4', 'time_shift', 4))
        with self.assertRaises(ValueError):
            parse_condition('time_shift_0')


if __name__ == '__main__':
    unittest.main()

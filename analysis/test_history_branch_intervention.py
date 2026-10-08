import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from analysis.history_branch_intervention_eval import InterventionLSCLIFNeuron, parse_condition
from analysis.prefix_mask_spike_similarity import load_namespace_args
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

    def test_lsclif_zero_removes_only_ls_history(self):
        neuron = InterventionLSCLIFNeuron(tau=2.0, history_weight=1.0, history_power=0.0)
        neuron.set_history_intervention('zero')
        spike = neuron(torch.tensor([[0.6]]))
        self.assertEqual(float(spike.item()), 0.0)
        self.assertAlmostEqual(float(neuron.n.item()), 0.6, places=6)
        self.assertAlmostEqual(float(neuron.v.item()), 0.6, places=6)
        self.assertAlmostEqual(float(neuron.m.item()), 0.0, places=6)

    def test_300_epoch_args_log_uses_serialized_namespace(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'args.txt'
            epoch_logs = '\n'.join(f'epoch={epoch}, train_loss=1.0' for epoch in range(300))
            path.write_text(
                "Namespace(model='spiking_vgg11_bn', neuron_model='LSCLIF', epochs=300)\n"
                + epoch_logs,
                encoding='utf-8',
            )
            args = load_namespace_args(path)
        self.assertEqual(args.neuron_model, 'LSCLIF')
        self.assertEqual(args.epochs, 300)


if __name__ == '__main__':
    unittest.main()

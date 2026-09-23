import unittest

import torch

from modules.neuron import LSRPLIFNeuron
from modules.neuron import RPLIFNeuron, VanillaLIFNeuron
from models.spiking_resnet import spiking_resnet18
from models.spiking_vgg_bn import spiking_vgg11_bn


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


class RPLIFLIFHeadTest(unittest.TestCase):
    def test_only_final_resnet_neuron_is_replaced(self):
        model = spiking_resnet18(
            neuron=RPLIFNeuron,
            rplif_lif_head=True,
            rplif_lif_head_neuron=VanillaLIFNeuron,
        )

        self.assertIsInstance(model.relu1, VanillaLIFNeuron)
        self.assertIsInstance(model.layer1[0].relu1, RPLIFNeuron)
        self.assertIsInstance(model.layer4[-1].relu2, RPLIFNeuron)

    def test_default_keeps_final_rplif_neuron(self):
        model = spiking_resnet18(neuron=RPLIFNeuron)
        self.assertIsInstance(model.relu1, RPLIFNeuron)

    def test_only_final_vgg_neuron_is_replaced(self):
        model = spiking_vgg11_bn(
            neuron=RPLIFNeuron,
            rplif_lif_head=True,
            rplif_lif_head_neuron=VanillaLIFNeuron,
        )
        neurons = [
            module for module in model.modules()
            if isinstance(module, (RPLIFNeuron, VanillaLIFNeuron))
        ]
        self.assertTrue(all(isinstance(module, RPLIFNeuron) for module in neurons[:-1]))
        self.assertIsInstance(neurons[-1], VanillaLIFNeuron)


if __name__ == '__main__':
    unittest.main()

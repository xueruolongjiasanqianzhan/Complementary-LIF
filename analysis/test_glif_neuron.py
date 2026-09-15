import unittest

import torch

from modules.neuron import GLIFNeuron, LSGLIFNeuron


class GLIFNeuronTest(unittest.TestCase):
    def test_gate_initialization_and_single_step_dynamics(self):
        neuron = GLIFNeuron(
            tau=2.0,
            v_threshold=1.0,
            glif_alpha=0.5,
            glif_beta=0.8,
            glif_gamma=0.75,
        )
        output = neuron(torch.tensor([0.5]))

        self.assertEqual(output.item(), 0.0)
        self.assertAlmostEqual(neuron.decay_gate.item(), 0.5, places=6)
        self.assertAlmostEqual(neuron.input_gate.item(), 0.8, places=6)
        self.assertAlmostEqual(neuron.reset_gate.item(), 0.75, places=6)
        self.assertAlmostEqual(neuron.v.item(), 0.4, places=6)

    def test_soft_reset_and_reset_method(self):
        neuron = GLIFNeuron(
            glif_alpha=0.5,
            glif_beta=0.8,
            glif_gamma=0.75,
            v_threshold=1.0,
        )
        output = neuron(torch.tensor([2.0]))

        self.assertEqual(output.item(), 1.0)
        self.assertAlmostEqual(neuron.v.item(), 0.85, places=5)
        neuron.reset()
        self.assertIsNone(neuron.v)

    def test_gate_parameters_receive_gradients(self):
        neuron = GLIFNeuron(glif_alpha=0.5, glif_beta=0.8, glif_gamma=0.75)
        loss = sum(neuron(torch.tensor([value], requires_grad=True)).sum() for value in (0.4, 0.7, 1.1))
        loss.backward()

        self.assertIsNotNone(neuron.alpha.grad)
        self.assertIsNotNone(neuron.beta.grad)
        self.assertIsNotNone(neuron.gamma.grad)
        self.assertTrue(torch.isfinite(neuron.alpha.grad))
        self.assertTrue(torch.isfinite(neuron.beta.grad))
        self.assertTrue(torch.isfinite(neuron.gamma.grad))

    def test_rejects_invalid_gate_initialization(self):
        with self.assertRaisesRegex(ValueError, 'glif_beta'):
            GLIFNeuron(glif_beta=1.0)


class LSGLIFNeuronTest(unittest.TestCase):
    def test_history_branch_changes_firing_decision(self):
        glif = GLIFNeuron(glif_alpha=0.5, glif_beta=0.8, glif_gamma=0.75)
        lsglif = LSGLIFNeuron(
            glif_alpha=0.5,
            glif_beta=0.8,
            glif_gamma=0.75,
            history_weight=1.0,
            history_power=0.0,
        )

        x = torch.tensor([0.7])
        self.assertEqual(glif(x).item(), 0.0)
        self.assertEqual(lsglif(x).item(), 1.0)
        self.assertAlmostEqual(lsglif.n.item(), 0.56, places=5)

    def test_reset_clears_both_membranes(self):
        neuron = LSGLIFNeuron()
        neuron(torch.ones(2))
        neuron.reset()

        self.assertIsNone(neuron.v)
        self.assertIsNone(neuron.n)
        self.assertEqual(neuron.step_count, 0)


if __name__ == '__main__':
    unittest.main()

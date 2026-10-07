"""MSF integration checks; run from the repository root with unittest."""
import argparse
import ast
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

import torch
from spikingjelly.clock_driven import functional

from models.spiking_resnet import spiking_resnet18
from models.spiking_vgg_bn import spiking_vgg11_bn
from modules import neuron


ROOT = Path(__file__).resolve().parents[1]


def cli_namespace(filename, arguments):
    """Execute the entry point's actual parser without loading datasets or CUDA."""
    tree = ast.parse((ROOT / filename).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    parser_nodes = []
    for node in main.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == 'args' for target in node.targets):
            break
        parser_nodes.append(node)
    namespace = {'argparse': argparse}
    exec(compile(ast.Module(body=parser_nodes, type_ignores=[]), filename, 'exec'), namespace)
    return namespace['parser'].parse_args(arguments), main


def cli_neuron(filename, arguments):
    args, main = cli_namespace(filename, arguments)
    scope = {'args': args, 'neuron': neuron, 'surrogate_function': neuron.Rectangle()}
    for node in main.body:
        # Exercise the actual neuron dispatch and kwargs passed to all backbones.
        if isinstance(node, ast.If) and 'neuron_model' in ast.unparse(node.test):
            if any(isinstance(child, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'neuron_model'
                    for target in child.targets) for child in ast.walk(node)):
                exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), scope)
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == 'neuron_kwargs' for target in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), scope)
    return args, scope['neuron_model'], scope['neuron_kwargs']


class MSFDynamicsTests(unittest.TestCase):
    def test_arbitrary_D_and_threshold_equality(self):
        for D in (1, 2, 4, 7):
            with self.subTest(D=D):
                model = neuron.MSFNeuron(msf_D=D)
                x = torch.tensor([-1., 0., 0.999, 1., 2., 4., 100.])
                self.assertEqual(model(x).tolist(), [0, 0, 0, 1, min(2, D), min(4, D), D])
                self.assertEqual(list(model.parameters()), [])
                self.assertEqual(model.thresholds.tolist(), list(range(1, D + 1)))

    def test_strict_surrogate_window_and_overlap_sum(self):
        # alpha=1 overlaps the windows for thresholds 1 and 2.
        model = neuron.MSFNeuron(msf_D=2, msf_alpha=1.)
        x = torch.tensor([0., 0.5, 1., 1.5, 2., 2.5, 3.], requires_grad=True)
        scales = torch.arange(1., 8.)
        (model(x) * scales).sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([0., .5, .5, 1., .5, .5, 0.]) * scales)
        # Official default endpoints are excluded too.
        model = neuron.MSFNeuron(msf_D=1)
        x = torch.tensor([.5, .75, 1., 1.25, 1.5], requires_grad=True)
        model(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([0., 1., 1., 1., 0.]))

    def test_leak_and_boolean_reset_for_every_level(self):
        model = neuron.MSFNeuron()
        model(torch.tensor([.8, 1., 2., 3., 4.], requires_grad=True))
        torch.testing.assert_close(model.v, torch.tensor([.8, 0., 0., 0., 0.]))
        model(torch.zeros(5))
        torch.testing.assert_close(model.v, torch.tensor([.2, 0., 0., 0., 0.]))

    def test_reset_mask_does_not_add_surrogate_reset_gradient(self):
        model = neuron.MSFNeuron()
        x = torch.tensor([.8, 1.2, 2.2], requires_grad=True)
        model(x)
        model.v.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([1., 0., 0.]))

    def test_D1_matches_manual_single_threshold_hard_reset(self):
        model = neuron.MSFNeuron(msf_D=1, msf_decay=.4, msf_threshold=1.2)
        state = torch.zeros(3)
        for x in (torch.tensor([.8, 1.2, -1.]), torch.tensor([.7, 0., 2.]), torch.zeros(3)):
            membrane = .4 * state + x
            expected = (membrane >= 1.2).float()
            state = torch.where(expected.bool(), 0., membrane)
            torch.testing.assert_close(model(x), expected)
            torch.testing.assert_close(model.v, state)

    def test_default_mechanism_example(self):
        for cls, expected in ((neuron.MSFNeuron, [0, 1, 0, 0]),
                              (neuron.LSMSFNeuron, [1, 2, 0, 0])):
            model = cls()
            self.assertEqual([model(torch.tensor([x])).item() for x in (.8, 1.6, .4, 0.)], expected)

    def test_generic_parameters_do_not_override_msf(self):
        class ForbiddenSurrogate:
            def __call__(self, _):
                raise AssertionError('MSF must use its own surrogate')
        for cls in (neuron.MSFNeuron, neuron.LSMSFNeuron):
            model = cls(tau=100., decay_input=True, v_threshold=99., v_reset=5.,
                        surrogate_function=ForbiddenSurrogate(), detach_reset=False)
            model(torch.tensor([1.2]))
            self.assertEqual(model.v.item(), 0.)
            self.assertEqual(model.msf_decay, .25)
            self.assertEqual(model.thresholds[0].item(), 1.)

    def test_invalid_options_and_unsupported_combinations(self):
        for cls in (neuron.MSFNeuron, neuron.LSMSFNeuron):
            for options in ({'msf_D': 0}, {'msf_D': 1.5}, {'msf_D': True},
                            {'msf_alpha': 0}, {'msf_alpha': float('nan')},
                            {'msf_threshold': -1}, {'msf_decay': 1.1}):
                with self.subTest(cls=cls, options=options), self.assertRaises(ValueError):
                    cls(**options)
            for flag in ('asn_enable', 'success_modulation_enable', 'synaptic_release_enable',
                         'multiple_step', 'rplif_lif_head'):
                with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, flag):
                    cls(**{flag: True})
            with self.assertRaises(TypeError):
                cls()(torch.tensor([1]))


class LSMSFHistoryTests(unittest.TestCase):
    def test_history_survives_firing_and_all_states_reset(self):
        model = neuron.LSMSFNeuron()
        model(torch.tensor([.8]))
        self.assertEqual(model.v.item(), 0.)
        torch.testing.assert_close(model.n, torch.tensor([.8]))
        model(torch.tensor([1.6]))
        torch.testing.assert_close(model.n, torch.tensor([1.8]))
        self.assertTrue(model.has_fired.item())
        functional.reset_net(model)
        self.assertIsNone(model.v)
        self.assertIsNone(model.n)
        self.assertIsNone(model.has_fired)
        self.assertEqual(model.step_count, 0)
        self.assertEqual(model(torch.tensor([.8])).item(), 1)

    def test_nonfiring_state_keeps_main_not_total_membrane(self):
        model = neuron.LSMSFNeuron()
        self.assertEqual(model(torch.tensor([.3])).item(), 0.)
        torch.testing.assert_close(model.v, torch.tensor([.3]))
        torch.testing.assert_close(model.n, torch.tensor([.3]))

    def test_zero_fixed_history_matches_msf_outputs_and_input_gradients(self):
        inputs = torch.tensor([[.8, .2], [1.6, .7], [.4, 2.1], [0., .3]])
        outputs, gradients = [], []
        for model in (neuron.MSFNeuron(), neuron.LSMSFNeuron(history_weight=0.)):
            x = inputs.clone().requires_grad_()
            y = torch.stack([model(step) for step in x])
            (y * torch.arange(1., 9.).reshape(4, 2)).sum().backward()
            outputs.append(y.detach())
            gradients.append(x.grad)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)

    def test_history_gradient_crosses_a_main_reset(self):
        model = neuron.LSMSFNeuron(msf_alpha=1.)
        first = torch.tensor([.8], requires_grad=True)
        model(first)
        model(torch.tensor([.4])).sum().backward()
        self.assertGreater(first.grad.item(), 0.)

    def test_post_spike_uses_prior_firing_and_accumulates_before_first_spike(self):
        model = neuron.LSMSFNeuron(history_mode='post_spike')
        self.assertEqual(model(torch.tensor([.8])).item(), 0)
        self.assertEqual(model(torch.tensor([.8])).item(), 1)  # main=1, no LS yet
        self.assertEqual(model(torch.tensor([.95])).item(), 1)  # LS enabled now

    def test_half_and_missing_layer_metadata_match_standard_lslif(self):
        for cls in (neuron.LSLIFNeuron, neuron.LSMSFNeuron):
            for index, expected in ((0, 'post_spike'), (1, 'post_spike'), (2, 'all'), (4, 'all')):
                self.assertEqual(cls(history_mode='half', layer_index=index, total_layers=5).history_mode,
                                 expected)
            self.assertEqual(cls(history_mode='half').history_mode, 'all')

    def test_fixed_weight_is_not_clipped(self):
        model = neuron.LSMSFNeuron(history_weight=1., history_weight_lo=-.8, history_weight_hi=.8)
        self.assertEqual(model._get_history_weight(torch.float32, torch.device('cpu')).item(), 1.)
        self.assertEqual(list(model.parameters()), [])

    def test_learnable_parameters_match_lslif_and_receive_gradients(self):
        for bounds in ({}, {'history_weight_lo': -.8, 'history_weight_hi': .8}):
            options = dict(history_weight=.3, history_power=.6, history_learn_weight=True,
                           history_learn_power=True, history_weight_per_step=True, history_max_steps=2,
                           **bounds)
            model = neuron.LSMSFNeuron(msf_alpha=2., **options)
            reference = neuron.LSLIFNeuron(**options)
            for name, value in reference.named_parameters():
                torch.testing.assert_close(dict(model.named_parameters())[name], value)
            for step in (1, 2, 3):
                torch.testing.assert_close(model._get_history_weight(torch.float32, torch.device('cpu'), step),
                                           reference._get_history_weight(torch.float32, torch.device('cpu'), step))
            y = [model(torch.tensor([.4])) for _ in range(3)]
            torch.stack(y).sum().backward()
            self.assertTrue(torch.isfinite(model.history_weight_raw.grad).all())
            self.assertTrue((model.history_weight_raw.grad.abs() > 0).all())
            self.assertGreater(model.history_power_raw.grad.abs().item(), 0.)
            with torch.no_grad():
                model.history_weight_raw.copy_(torch.tensor([-.5, .5]))
            torch.testing.assert_close(model._get_history_weight(torch.float32, torch.device('cpu'), 2),
                                       model._get_history_weight(torch.float32, torch.device('cpu'), 10))
            parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
            model.reset()
            for name, value in model.named_parameters():
                torch.testing.assert_close(value, parameters[name])


class MSFIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_shapes_dtype_autocast_and_new_sequence(self):
        for cls in (neuron.MSFNeuron, neuron.LSMSFNeuron):
            for shape in ((), (3,), (2, 3), (2, 3, 4, 4)):
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    model = cls()
                    x = torch.full(shape, .8, dtype=dtype, requires_grad=True)
                    with torch.autocast('cpu', dtype=torch.bfloat16):
                        y = model(x)
                    self.assertEqual(y.dtype, dtype)
                    self.assertEqual(y.shape, x.shape)
                    self.assertEqual(model.v.dtype, torch.float32)
                    y.float().sum().backward()
                    self.assertTrue(torch.isfinite(x.grad).all())
                    model(torch.ones(5))  # shape change starts a new sequence
                    if cls is neuron.LSMSFNeuron:
                        self.assertEqual(model.step_count, 1)
                    model.reset()
                    self.assertIsNone(model.v)

    def test_resnet_vgg_multistep_backward_reset_and_half_modes(self):
        for factory in (spiking_resnet18, spiking_vgg11_bn):
            for cls in (neuron.MSFNeuron, neuron.LSMSFNeuron):
                with self.subTest(factory=factory.__name__, cls=cls.__name__):
                    torch.manual_seed(23)
                    channels = 2 if factory is spiking_vgg11_bn else 3
                    model = factory(neuron=cls, num_classes=10, history_mode='half', c_in=channels)
                    x = torch.randn(2, channels, 32, 32, requires_grad=True)
                    with torch.autocast('cpu', dtype=torch.bfloat16):
                        y = sum(model(x) for _ in range(3))
                        loss = torch.nn.functional.cross_entropy(y.float(), torch.tensor([1, 2]))
                    self.assertEqual(y.shape, (2, 10))
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(torch.isfinite(x.grad).all())
                    self.assertGreater(x.grad.abs().sum().item(), 0.)
                    layers = [m for m in model.modules() if isinstance(m, cls)]
                    if cls is neuron.LSMSFNeuron:
                        modes = [m.history_mode for m in layers]
                        self.assertEqual(modes[:len(modes)//2], ['post_spike'] * (len(modes)//2))
                        self.assertEqual(modes[len(modes)//2:], ['all'] * (len(modes)-len(modes)//2))
                    functional.reset_net(model)
                    self.assertTrue(all(m.v is None for m in layers))
                    if cls is neuron.LSMSFNeuron:
                        self.assertTrue(all(m.n is None and m.has_fired is None and m.step_count == 0 for m in layers))
                    # Repeating the same batch after reset produces the same first-step output.
                    model.eval()
                    with torch.no_grad():
                        first = model(x)
                        functional.reset_net(model)
                        second = model(x)
                    torch.testing.assert_close(first, second, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU unavailable; GPU AMP is unverified')
    def test_cuda_device_and_amp(self):
        for cls in (neuron.MSFNeuron, neuron.LSMSFNeuron):
            model = cls().cuda()
            x = torch.tensor([.8, 1.6], device='cuda', requires_grad=True)
            with torch.autocast('cuda'):
                y = model(x)
            y.sum().backward()
            self.assertEqual(y.device, x.device)
            self.assertTrue(torch.isfinite(x.grad).all())
            model.cpu()
            model(torch.tensor([.8, 1.6]))
            self.assertEqual(model.v.device.type, 'cpu')
            if cls is neuron.LSMSFNeuron:
                self.assertEqual(model.step_count, 1)

    def test_training_inference_dispatch_and_config_checkpoint_roundtrip(self):
        for filename in ('train.py', 'inference.py'):
            for name in ('MSF', 'LSMSF'):
                arguments = ['-neuron_model', name, '-msf_D', '3', '-msf_threshold', '1.2',
                             '-msf_decay', '.3', '-msf_alpha', '.8', '-T', '3', '-tau', '99',
                             '-v_threshold', '.1', '-history_weight', '.3', '-history_power', '.7',
                             '-history_learn_weight', '-history_weight_per_step', '-history_learn_power']
                args, cls, kwargs = cli_neuron(filename, arguments)
                original = cls(**kwargs)
                config = neuron.msf_experiment_config(args)
                payload = io.BytesIO()
                original(torch.tensor([.7, 1.4]))  # runtime state must not be checkpointed
                torch.save({'net': original.state_dict(), 'neuron_config': config,
                            'experiment_args': vars(args)}, payload)
                payload.seek(0)
                checkpoint = torch.load(payload, weights_only=True)
                saved = json.loads(json.dumps(checkpoint['neuron_config']))
                rebuilt = getattr(neuron, saved['neuron_model'] + 'Neuron')(**saved['kwargs'])
                rebuilt.load_state_dict(checkpoint['net'], strict=True)
                self.assertIsNone(rebuilt.v)
                original.reset()
                x1 = torch.tensor([.7, 1.4], requires_grad=True)
                x2 = x1.detach().clone().requires_grad_()
                output1 = original(x1)
                output2 = rebuilt(x2)
                torch.testing.assert_close(output1, output2, rtol=0, atol=0)
                output1.sum().backward()
                output2.sum().backward()
                torch.testing.assert_close(x1.grad, x2.grad, rtol=0, atol=0)
                torch.testing.assert_close(original.v, rebuilt.v, rtol=0, atol=0)
                self.assertEqual(original.msf_decay, .3)
                self.assertEqual(original.thresholds[0].item(), torch.tensor(1.2).item())

    def test_run_tag_distinguishes_all_ls_settings(self):
        args, _ = cli_namespace('train.py', ['-neuron_model', 'LSMSF'])
        first = neuron.msf_run_tag(neuron.msf_experiment_config(args))
        args.history_eps = .001
        second = neuron.msf_run_tag(neuron.msf_experiment_config(args))
        self.assertNotEqual(first, second)
        self.assertLess(len(second), 100)

    def test_real_cli_help_and_unsupported_flags_fail_before_dataset_loading(self):
        for filename in ('train.py', 'inference.py'):
            result = subprocess.run([sys.executable, filename, '--help'], cwd=ROOT,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('-msf_D', result.stdout)
            self.assertIn('LSMSF', result.stdout)
            result = subprocess.run([sys.executable, filename, '-neuron_model', 'LSMSF',
                                     '-synaptic_release_enable'], cwd=ROOT,
                                    capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('synaptic_release_enable is not supported with pure MSF/LSMSF', result.stderr)

    def test_five_dataset_recipes_parse_with_matching_msf_settings(self):
        import shlex
        text = (ROOT / 'neuron_training_commands_by_dataset.txt').read_text().replace('\\\n', ' ')
        pairs = {}
        for line in text.splitlines():
            if line.startswith('python train.py ') and any(
                    f'-neuron_model {name}' in line for name in ('MSF', 'LSMSF')):
                args, _ = cli_namespace('train.py', shlex.split(line)[2:])
                pairs.setdefault(args.dataset, {})[args.neuron_model] = args
        self.assertEqual(set(pairs), {'cifar10', 'cifar100', 'tiny_imagenet', 'DVSCIFAR10', 'dvsgesture'})
        for pair in pairs.values():
            a, b = vars(pair['MSF']), vars(pair['LSMSF'])
            for key in a:
                if key != 'neuron_model':
                    self.assertEqual(a[key], b[key], key)


if __name__ == '__main__':
    unittest.main()

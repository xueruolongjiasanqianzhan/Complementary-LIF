import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from analysis.analyze_temporal_gradient import (
    _final_step_retention_matrix,
    _display_cmap,
    _install_numpy_legacy_aliases,
    _time_frame,
    _validate_pair,
    build_parser,
    evenly_spaced_indices,
    load_config,
    parse_namespace,
)


class TemporalGradientAnalysisTests(unittest.TestCase):
    def test_parse_namespace_is_safe_and_supports_historical_values(self):
        config = parse_namespace(
            "Namespace(T=16, dataset='DVSCIFAR10', data_dir='/tmp/中文', "
            "enabled=True, upper=None)\nepoch=302")
        self.assertEqual(config["T"], 16)
        self.assertEqual(config["data_dir"], "/tmp/中文")
        self.assertTrue(config["enabled"])
        self.assertIsNone(config["upper"])

    def test_load_config_reads_namespace_from_resumed_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "args.txt").write_text(
                "Namespace(T=16, dataset='DVSCIFAR10', model='spiking_vgg11_bn')\n"
                "epoch=302, test_acc=0.716\n", encoding="utf-8")
            self.assertEqual(load_config(path)["model"], "spiking_vgg11_bn")

    def test_evenly_spaced_indices_include_both_ends(self):
        self.assertEqual(evenly_spaced_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(evenly_spaced_indices(3, 10), [0, 1, 2])

    def test_pair_validation_rejects_different_time_steps(self):
        common = {"dataset": "DVSCIFAR10", "model": "spiking_vgg11_bn"}
        with self.assertRaisesRegex(ValueError, "same T"):
            _validate_pair({**common, "T": 16}, {**common, "T": 8})

    def test_pair_validation_warns_about_weight_decay_mismatch(self):
        common = {"dataset": "DVSCIFAR10", "model": "spiking_vgg11_bn", "T": 16}
        with self.assertWarnsRegex(UserWarning, "weight_decay"):
            _validate_pair(
                {**common, "weight_decay": 5e-5},
                {**common, "weight_decay": 5e-4})

    def test_cli_defaults_to_middle_vgg_layer(self):
        args = build_parser().parse_args(["--ls-run", "ls", "--baseline-run", "base"])
        self.assertEqual(args.layer, "layer3.6")
        self.assertEqual(args.checkpoint_name, "checkpoint_max.pth")
        self.assertEqual(args.max_neurons, 512)
        self.assertEqual(args.gradient_target, "final")
        self.assertEqual(args.gradient_source, "input")
        self.assertEqual(args.aggregation, "batch-mean-abs")
        self.assertEqual(args.normalization, "final-step")
        self.assertEqual(args.color_scale, "symlog")
        self.assertEqual(args.normalized_color_gamma, 0.35)
        self.assertEqual(args.difference_linthresh, 0.02)
        self.assertEqual(args.fig_width, 21.0)

    def test_script_can_start_outside_repository(self):
        script = Path(__file__).with_name("analyze_temporal_gradient.py").resolve()
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(script), "--help"], cwd=directory,
                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--ls-run", result.stdout)

    def test_numpy_legacy_aliases_are_installed_before_dataset_imports(self):
        fake_numpy = SimpleNamespace(sctypeDict={"int": int})
        _install_numpy_legacy_aliases(fake_numpy)
        self.assertIs(fake_numpy.int, int)
        self.assertIs(fake_numpy.bool, bool)
        self.assertIs(fake_numpy.object, object)
        self.assertIs(fake_numpy.typeDict, fake_numpy.sctypeDict)

    def test_time_frame_accepts_default_dvs_list_batch(self):
        frames = ["t0", "t1", "t2"]
        self.assertEqual(_time_frame(frames, 1, 3), "t1")
        with self.assertRaisesRegex(ValueError, "Expected 4 frame tensors"):
            _time_frame(frames, 1, 4)

    def test_normalized_heatmap_uses_white_to_blue_colormap(self):
        self.assertEqual(_display_cmap("per-neuron"), "Blues")
        self.assertEqual(_display_cmap("none"), "RdBu_r")

    def test_final_step_matrix_normalization_uses_orders_of_magnitude(self):
        import numpy as np

        matrix = np.asarray([[1.0, 10.0, 100.0], [0.001, 0.01, 0.1]])
        retention = _final_step_retention_matrix(matrix, np)
        np.testing.assert_allclose(retention[:, -1], np.ones(2))
        np.testing.assert_allclose(retention[0], [0.01, 0.1, 1.0])
        np.testing.assert_allclose(np.diff(np.log10(retention[0])), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

from analysis.analyze_temporal_gradient import (
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

    def test_cli_defaults_to_middle_vgg_layer(self):
        args = build_parser().parse_args(["--ls-run", "ls", "--baseline-run", "base"])
        self.assertEqual(args.layer, "layer3.6")
        self.assertEqual(args.checkpoint_name, "checkpoint_max.pth")
        self.assertEqual(args.max_neurons, 512)

    def test_script_can_start_outside_repository(self):
        script = Path(__file__).with_name("analyze_temporal_gradient.py").resolve()
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(script), "--help"], cwd=directory,
                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--ls-run", result.stdout)


if __name__ == "__main__":
    unittest.main()

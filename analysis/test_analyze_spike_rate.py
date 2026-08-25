import csv
import json
import tempfile
import unittest
from pathlib import Path

from analysis.analyze_spike_rate import _parse_layers, load_run, main, select_window


LAYERS = {
    "layer1.0.relu1": 0.02,
    "layer3.0.relu1": 0.10,
    "layer4.1.relu2": 0.15,
}


def write_run(
    path: Path,
    neuron: str,
    epochs: list[tuple[int, float, float]],
    historical_max=None,
    ordered_dict_entries=False,
):
    path.mkdir()
    (path / "run_summary.json").write_text(json.dumps({
        "dataset": "cifar10", "model": "spiking_resnet18", "neuron_model": neuron,
        "seed": 2022, "time_steps": 4, "batch_size": 128,
    }))
    with (path / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "test_acc", "max_test_acc", "test_spike_rate_global"])
        running = historical_max or 0.0
        for epoch, accuracy, rate in epochs:
            running = max(running, accuracy)
            writer.writerow([epoch, accuracy, running, rate])
    lines = []
    for epoch, accuracy, rate in epochs:
        running = max(historical_max or 0.0, max(a for e, a, r in epochs if e <= epoch))
        layer_payload = list(LAYERS.items()) if ordered_dict_entries else LAYERS
        lines.append(
            f"epoch={epoch}, train_loss=1, train_acc=.5, test_loss=1, test_acc={accuracy}, "
            f"max_test_acc={running}, total_time=1, test_spike_rate_global={rate}, "
            "escape_time=2026-01-01 00:00:00\n"
            f"test_spike_rate_layers=OrderedDict({layer_payload!r})\n"
        )
    (path / "args.txt").write_text("".join(lines))


class AnalyzeSpikeRateTest(unittest.TestCase):
    def test_parses_plain_dictionary(self):
        parsed = _parse_layers(
            f"test_spike_rate_layers={LAYERS!r}\n",
            Path("args.txt"),
            1,
        )
        self.assertEqual(parsed, LAYERS)

    def test_parses_legacy_ordered_dict_entry_list(self):
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory) / "run"
            write_run(
                run_path,
                "LSLIF",
                [(302, .716, .07129568747214012), (303, .709, .07292803863433746)],
                historical_max=.725,
                ordered_dict_entries=True,
            )
            run = load_run(run_path)
            self.assertEqual(run.records[302].layer_rates, LAYERS)
            self.assertAlmostEqual(run.records[303].global_rate, .07292803863433746)

    def test_resume_uses_best_observed_accuracy_and_one_sided_window(self):
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory) / "run"
            write_run(run_path, "LSLIF", [(51, .80, .10), (52, .79, .11), (53, .78, .12)], historical_max=.90)
            run = load_run(run_path)
            best, window = select_window(run, 3)
            self.assertEqual(best.epoch, 51)
            self.assertEqual([record.epoch for record in window], [51, 52, 53])
            self.assertTrue(any("Historical max_test_acc" in warning for warning in run.warnings))

    def test_cli_writes_comparison_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ls_path, baseline_path, output = root / "ls", root / "baseline", root / "output"
            epochs = [(1, .60, .12), (2, .70, .10), (3, .65, .11)]
            write_run(ls_path, "LSLIF", epochs)
            write_run(baseline_path, "LIF", [(e, a - .02, r + .02) for e, a, r in epochs])
            result = main([
                "--ls-run", str(ls_path), "--baseline-run", str(baseline_path),
                "--output-dir", str(output), "--window-size", "3",
            ])
            self.assertEqual(result, 0)
            self.assertTrue((output / "mean_spike_rate_comparison.png").is_file())
            self.assertTrue((output / "spike_rate_summary.csv").is_file())
            self.assertTrue((output / "spike_rate_comparison.csv").is_file())
            self.assertTrue((output / "mean_spike_rate_comparison.png").read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()

# LS / non-LS firing-rate comparison

`analysis/analyze_spike_rate.py` compares two training run directories around
each run's best **observed** test-accuracy epoch. It reads the existing
`metrics.csv`, `args.txt`, and `run_summary.json` files.
Both historical `OrderedDict([('layer', rate), ...])` records and newer
`OrderedDict({'layer': rate})`/plain-dictionary records are supported.

Install the plotting dependency once before running the tool:

```bash
python -m pip install -r analysis/requirements.txt
```

## Usage

```bash
python analysis/analyze_spike_rate.py \
  --ls-run ./logs/lslif_run \
  --baseline-run ./logs/lif_run \
  --output-dir ./spike_rate_comparison
```

The default five-epoch window is centered on the best observed epoch and is
shifted to one side at a log boundary. This is important for resumed runs: the
script keeps the real epoch numbers, analyzes only records present in the given
directory, and does not use a historical `max_test_acc` whose corresponding
firing rates may be in an earlier directory.

For ResNet logs, the default representative layers are:

- shallow: `layer1.0.relu1`;
- middle: `layer3.0.relu1`;
- deep: `layer4.1.relu2`.

For other backbones, the script chooses layers near 25%, 50%, and 75% of the
common recorded layer order. Override either behavior explicitly when needed:

```bash
python analysis/analyze_spike_rate.py \
  --ls-run ./logs/ls_run \
  --baseline-run ./logs/non_ls_run \
  --window-size 5 \
  --shallow-layer layer1.0.relu1 \
  --middle-layer layer3.0.relu1 \
  --deep-layer layer4.1.relu2
```

`--window-size` must be a positive odd integer. All three layer overrides must
be supplied together and must exist in both selected windows.

## Outputs

The output directory contains:

- `mean_spike_rate_comparison.png`: grouped mean-rate bars for global,
  shallow, middle, and deep scopes. The high-resolution chart identifies the
  compared neuron models and exact bar values; exact representative layer names
  remain available in the CSV output without crowding the x-axis;
- `mean_spike_rate_comparison.svg`: the same chart as a resolution-independent
  vector image. Prefer this file for papers, slides, or arbitrary resizing;
- `spike_rate_summary.csv`: run metadata, selected epochs, representative
  layers, and mean rates;
- `spike_rate_comparison.csv`: LS-minus-baseline absolute and relative rate
  changes.

The script also reports configuration mismatches, partial/resumed logs,
historical maxima that are unavailable in the supplied directory, one-sided
windows, severe accuracy changes inside a window, and exact duplicate epoch
records. Conflicting duplicate records are rejected rather than overwritten.

## Changing image size

Image size follows Matplotlib's standard `figure size × DPI` rule. The defaults
are 14 × 8 inches at 300 DPI, producing an approximately 4200 × 2400 PNG. Set
the dimensions directly from the command line:

```bash
python analysis/analyze_spike_rate.py \
  --ls-run ./logs/ls_run \
  --baseline-run ./logs/non_ls_run \
  --fig-width 12 \
  --fig-height 7 \
  --dpi 400
```

That produces an approximately 4800 × 2800 PNG. `bbox_inches="tight"` may trim
some outer whitespace, so exact pixel dimensions can differ slightly. The SVG
uses the same layout but remains sharp at every zoom level.

If editing the code directly is preferred, change the defaults of
`--fig-width`, `--fig-height`, and `--dpi` in `build_parser()`.

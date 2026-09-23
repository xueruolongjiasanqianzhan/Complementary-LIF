# Pre-threshold membrane provenance for LIF/LSLIF

This mechanism experiment measures how much of the membrane used for each
threshold decision comes from the current input and from each earlier input.
It does **not** treat reset loss as an additional source and does not make an
accuracy claim.

The script contains scalar LIF and LSLIF equations directly in the entry file.
Both receive the same deterministic input and reset naturally after
threshold-triggered spikes. Immediately before each threshold decision, the
source contributions sum to the simulated membrane.

After a soft reset, the observed residual main membrane is assigned back to
its pre-reset input sources in the same proportions. For example, if sources
`0.2` and `0.9` form a pre-reset membrane of `1.1` and reset leaves `0.1`, the
residual contributions are `0.1 * 2/11` and `0.1 * 9/11`. This proportional
provenance rule is an attribution convention: the neuron only defines the
total soft-reset result, not which input source lost each part of the
threshold. The LS source ledger is not reset.

## Run

Only NumPy and Matplotlib are required. From the repository root, run one file:

```bash
python analysis/analyze_membrane_sources.py \
  --out-dir analysis_results/membrane_source_attribution
```

The default deterministic input is:

```text
1.2 0.9 0 0 1.2 0.9 0 0 0 0
```

It creates natural charge/reset events followed by silent observation. A
custom sequence can be supplied with `--inputs`.

The two source-time heatmaps plot the absolute value of each signed source
percentage, while the CSV retains the original sign. They use a shared,
reversed power-law color scale with `--heatmap-gamma 0.35` by default: larger
magnitudes are darker and smaller magnitudes are lighter. The power transform
expands color differences among small late-time contributions without changing
their numerical magnitudes or the shared LIF/LSLIF scale. Use a smaller positive
value (for example `0.2`) for more low-value contrast, or `1.0` for a linear
color mapping:

```bash
python analysis/analyze_membrane_sources.py --heatmap-gamma 0.2
```

## Outputs

- `membrane_source_attribution.png` contains the actual pre-threshold membrane
  traces, LIF-versus-LSLIF historical shares, current/history stacked shares,
  and input-source-time heatmaps;
- `membrane_source_trace.csv` contains current/history contributions and
  counterfactual pre-threshold membranes for every decision step;
- `membrane_source_by_input_time.csv` contains every source-time contribution;
- `membrane_source_summary.json` reports spike counts, historical shares at
  LSLIF spike steps, LS-history-dependent threshold crossings, and exact
  reconstruction errors.

## Interpretation boundary

The primary quantity is the pre-threshold decision membrane. Historical share
is the fraction of that membrane descended from inputs at earlier time steps.
For LSLIF, historical contribution is separated into the resettable main path
and the non-reset LS path. A larger LS historical component shows that earlier
inputs continue to participate in a later firing decision through LS.

If the main membrane becomes negative after an LS-assisted spike, proportional
main-path contributions become signed. The values still sum exactly to the
real membrane, but they should be described as signed attributions rather than
probabilities. The default input is chosen to keep the primary visualization
simple. Retaining more historical membrane is a mechanism result only; whether
it helps a task must be tested separately.

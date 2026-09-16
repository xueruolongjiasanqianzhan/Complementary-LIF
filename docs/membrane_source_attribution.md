# LIF/LSLIF membrane-source attribution

This mechanism-only experiment asks how much of the membrane used for the
current firing decision comes from current input, past input, reset
subtractions, and the non-reset LS branch. It does not claim that retaining
more potential necessarily improves accuracy.

The experiment directly runs the repository's `VanillaLIFNeuron` and
`LSLIFNeuron` with the same deterministic input. Spikes and soft resets occur
naturally when each neuron's effective membrane reaches threshold; reset times
are not manually injected. A passive source ledger reconstructs the actual
membrane at every step and aborts if the reconstruction differs from the real
neuron state.

## Run

Use the normal training environment with PyTorch and SpikingJelly installed:

```bash
python analysis/analyze_membrane_sources.py \
  --out-dir analysis_results/membrane_source_attribution
```

The default input is a fixed charge/silence/recharge/silence sequence. A custom
deterministic sequence can be supplied, for example:

```bash
python analysis/analyze_membrane_sources.py \
  --inputs 0.6 0.6 0.6 0.6 0 0 0 0 \
  --history-weight 0.6 \
  --history-power 1.0
```

Outputs:

- `membrane_source_attribution.png`: signed LIF and LSLIF source stacks plus
  LSLIF absolute-contribution percentages;
- `membrane_source_trace.csv`: all per-step source terms and reconstruction
  errors;
- `membrane_source_summary.json`: configuration, reset counts, post-reset
  historical contributions, and maximum reconstruction errors.

## Interpretation

The signed decomposition keeps reset loss separate from input-derived state:

- `main current input`: the present input in the resettable membrane;
- `main past input`: decayed earlier inputs in the resettable membrane;
- `reset loss`: the negative, decayed contribution of earlier soft resets;
- `LS current input`: the LS branch's contribution from the present input;
- `LS past input`: earlier inputs retained through the non-reset LS branch.

The percentage panel normalizes absolute contribution magnitudes because reset
loss is negative and the net membrane can approach zero. It must be described
as a source-magnitude percentage, not an algebraic share of the final
membrane. The key mechanism result is a positive `LS past input` contribution
after a natural reset, especially during zero-input steps. At the same time,
the explicit negative `reset loss` remains visible. This supports the narrow
claim that LS preserves additional input-derived potential outside the reset
path; it does not claim that the reset subtraction itself becomes smaller.

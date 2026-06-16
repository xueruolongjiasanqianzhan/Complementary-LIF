# LSLIF ROX / delayed pulse experiment

`DH-SNN/delayed_xor` is treated as a vendored reference and is not modified.  The
LSLIF experiment lives in `analysis/lsl_if_delayed_xor.py` and mirrors the DH-SNN
delayed spiking XOR controls so the results remain comparable.

## Matched control variables

The script defaults match `DH-SNN/delayed_xor/delayed_xor_snn.py`:

| Control | Value |
| --- | --- |
| sequence length / delay | 200 time steps |
| high/low pulse rates | 0.6 / 0.2 |
| background noise rate | 0.01 |
| input channels | 20 |
| signal coding time | 10 steps |
| test-time pulse positions | 1 |
| hidden neurons | 16 |
| output classes | 2 |
| batch size | 500 |
| optimizer | Adam, `lr=1e-2` |
| scheduler | `StepLR(step_size=50, gamma=0.1)` |
| epochs / generated batches | 150 / 100 per epoch |
| loss window | only after the delayed pulse appears |

For strict comparability, the implementation also preserves the DH-SNN training
choice of applying `softmax` before `CrossEntropyLoss`, even though modern
PyTorch examples usually pass raw logits to `CrossEntropyLoss`.

## Outputs kept in parity with DH-SNN

The checked-in DH-SNN delayed-XOR script imports `matplotlib.pyplot`, but it does
not actually call `plt.plot` or `savefig`, so it does not generate figures.  To
avoid adding results that the reference experiment does not have, the LSLIF
script also does not generate plots or extra analysis artifact files.

The outputs are kept to the DH-SNN-style results:

- console logs for learning rate, train loss, and accuracy;
- an in-memory `acc_list` return from `train`;
- each forward pass returns the readout sequence `d2_output` with shape
  `[batch, time, 2]`;
- optional best-model checkpoint when `--save-best` is passed.

## Environment notes

The LSLIF script uses the current repository stack: Python 3, PyTorch, NumPy,
SpikingJelly, and this repository's `modules.neuron.LSLIFNeuron` /
`modules.surrogate.Rectangle`.  Install the runtime packages with:

```bash
pip install torch numpy spikingjelly
```

If your PyTorch build must match a specific CUDA version, install `torch` from
the official PyTorch command for your CUDA/CPU environment first, then install
`numpy spikingjelly`.

If you want to run the original DH-SNN file directly for side-by-side validation,
then install its extra imports (`tables`, `scipy`, `torchvision`, `matplotlib`) and
make sure the `SNN_layers` package is importable from `DH-SNN/delayed_xor`.
Those dependencies are not required for the LSLIF experiment added here because
we are not adding plots or DH-SNN-only functionality.

## Commands

See `analysis/lsl_if_delayed_xor_commands.txt` for the full LSLIF run, the local
vanilla LIF control, and a CPU smoke-test command.

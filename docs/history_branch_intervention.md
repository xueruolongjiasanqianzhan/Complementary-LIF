# LSLIF history-branch intervention experiment

This is the first, checkpoint-only experiment from the LS experiment roadmap.
It reuses a trained DVS-CIFAR10/VGG11 LSLIF checkpoint and the original test
set; it does not retrain or change checkpoint weights.

## Conditions

- `normal`: unchanged LSLIF inference;
- `zero`: replace the fused history term with zero;
- `shuffle`: roll history terms across batch samples, preserving their values
  while breaking sample correspondence;
- `time_shift_N`: use the history term from N steps earlier and emit zero for
  the first N steps.

The script currently targets the standard `LSLIFNeuron`. Other LS variants use
different forward equations and are intentionally excluded from this first
implementation rather than silently applying an incomplete intervention.

## Command

```bash
python analysis/history_branch_intervention_eval.py \
  --checkpoint /path/to/lslif/checkpoint_max.pth \
  --args /path/to/lslif/args.txt \
  --data-dir /path/to/DVS-CIFAR10 \
  --out-dir analysis_results/history_branch_intervention \
  --T 16 \
  --batch-size 16 \
  --conditions normal zero shuffle time_shift_1 time_shift_2 time_shift_4
```

The output directory contains:

- `history_branch_intervention_summary.csv`: accuracy and accuracy drop, NLL,
  true-class confidence, prediction-change rate, and both directions of
  correct/incorrect transitions relative to normal inference;
- `config.json`: checkpoint identity, conditions, number of intervened layers,
  and evaluation settings.

Interpret `normal > zero` as evidence that this checkpoint relies on the LS
branch. Requiring `normal > shuffle/time_shift` is stronger because it tests
whether sample-specific and time-aligned history matters, rather than merely
the presence of an extra membrane-scale term. These are paired interventions
on one trained model; a separately trained `history_weight=0` ablation remains
necessary to address train/test distribution shift.

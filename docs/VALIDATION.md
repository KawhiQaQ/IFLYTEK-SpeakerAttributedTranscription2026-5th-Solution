# Validation protocol

## Objective

The local protocol must estimate the deployable end-to-end system rather than
an oracle combination of fold-specific pipelines. The primary metric is
time-constrained minimum-permutation WER with a 5-second collar.

## Fixed folds

The 106 labeled development conversations are assigned to fixed folds using
session-level stratification. The split balances:

- conversation duration;
- number of reference speakers;
- speech and silence occupancy;
- overlap duration;
- number of annotated segments; and
- token count.

All segments from the same conversation stay in the same fold. No audio crop,
speaker identity, or segment from a validation conversation may enter the
corresponding training set.

## Promotion metric

Fold scores are pooled by error counts:

$$
\mathrm{tcpWER}_{\mathrm{pooled}}
=\frac{\sum_k E_k}{\sum_k N_k},
$$

where $E_k$ is the tcpWER error count and $N_k$ is the number of reference
tokens in fold $k$. The arithmetic mean of rounded fold-level rates is not
used for model selection.

The final system reports 1023 errors over 7904 reference tokens across the two
primary folds, corresponding to a pooled tcpWER of 0.129428.

## Fold-pure training

For every supervised component:

1. the validation fold is excluded before feature construction;
2. checkpoints and preprocessing statistics are fitted on training sessions
   only;
3. probability boundaries and epoch budgets are frozen using out-of-fold
   predictions; and
4. the deployment model is refit on all development sessions only after the
   architecture is fixed.

Public pretrained ASR and speaker encoders remain frozen unless a fold-specific
adaptation experiment explicitly states otherwise.

## Promotion criteria

A candidate is promoted only when:

- the pooled error count improves;
- the gain is not concentrated in one conversation;
- neither primary fold regresses materially;
- the test-time implementation matches the validated architecture; and
- no test label, pseudo-label, or test-derived target is used.

## Evaluation command

```bash
meeteval-wer tcpwer \
  -r path/to/ref.seglst.json \
  -h path/to/hyp.seglst.json \
  --collar 5
```

In addition to tcpWER, error analysis records lexical substitutions,
speaker-confusion errors, missed speech, false alarms, boundary errors, and
overlap-specific failures.

# Results and ablations

## Main results

| System | Pooled fold-0/1 tcpWER | Leaderboard tcpWER |
|---|---:|---:|
| Multi-ASR consensus + acoustic speaker graph | 0.14145 | 0.15396 |
| + WeSpeaker novel-speaker verification | 0.13120 | 0.14820 |
| + context-aware speaker Transformer | 0.13120 equivalent | 0.14743 |
| **+ overlap-consistency constraint** | **0.12943** | **0.14727** |

The final system ranks fifth on the competition leaderboard. Improvements are
reported cumulatively because each row preserves the transcription and speaker
modeling components above it.

## Context-aware speaker modeling

The contextual speaker Transformer replaces a purely local acoustic decision
with full-conversation reasoning. Its main benefit is not a large change in the
aggregate two-fold score; it improves the transfer from local validation to the
unlabeled evaluation distribution. The result supports using the Transformer
as a speaker-representation component rather than a test-time router.

## Overlap-consistency constraint

Words, timestamps, record ordering, and segment count are frozen. Only speaker
labels in physically impossible same-speaker overlaps of at least 0.20 seconds
may change.

| System | Fold 0 | Fold 1 | Pooled |
|---|---:|---:|---:|
| Context-aware model | 509/3845 | 528/4059 | 1037/7904 = 0.131197 |
| + overlap consistency | 496/3845 | 527/4059 | 1023/7904 = 0.129428 |

All three validation sessions affected by the rule improve; no validation
session regresses. Thresholds from 0.10 to 0.30 seconds produce the same pooled
score, whereas 0.05 seconds is rejected because fold 1 regresses.

On the test set, the constraint edits 11 speaker labels across 10 of 394
sessions. Material same-speaker overlaps decrease from 17 to 6. No test labels,
pseudo-labels, or test-time gradients are used.

## Rejected alternatives

- Short-turn A-B-A smoothing increased pooled tcpWER to 0.132212.
- Rare-speaker merging gave no gain or regressed.
- Cross-speaker contained-text deletion removed genuine backchannels.
- High-variance session routing concentrated its gain in one session and did
  not transfer to the leaderboard.
- Combining heterogeneous fold-specific pipelines produced an optimistic local
  estimate that did not correspond to a deployable test-time system.

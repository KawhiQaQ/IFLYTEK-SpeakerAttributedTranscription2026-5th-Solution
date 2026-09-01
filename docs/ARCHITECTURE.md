# CAST: Context-Aware Speaker-Attributed Transcription

## 1. Problem formulation

Let an input recording be a 16 kHz mono waveform $x$ containing a short,
free-form conversation. The system predicts a set of speaker-attributed
segments

$$
\mathcal{Y}=\{(t_i^{\mathrm{s}},t_i^{\mathrm{e}},r_i,\mathbf{w}_i)\}_{i=1}^{N},
$$

where $t_i^{\mathrm{s}}$ and $t_i^{\mathrm{e}}$ are segment boundaries,
$r_i$ is an anonymous speaker role, and $\mathbf{w}_i$ is a token sequence.
The evaluation metric is time-constrained minimum-permutation WER. Therefore,
lexical recognition, temporal localization, and speaker attribution must be
optimized jointly; improving any one of them by changing the others
unnecessarily can increase the final error.

## 2. System overview

Our system separates the problem into four coupled components:

1. heterogeneous ASR models generate complementary lexical and temporal views;
2. Sortformer activity tracks and pretrained speaker encoders form an acoustic
   speaker graph;
3. a context-aware speaker Transformer refines role assignments over the full
   conversation, while an independent WeSpeaker model verifies newly proposed
   roles; and
4. a deterministic consistency constraint removes physically impossible
   same-speaker overlaps without modifying lexical content or timestamps.

The design follows a conservative principle: each component changes only the
part of the hypothesis for which it has direct evidence. ASR consensus changes
tokens, acoustic models change speaker relations, and the final constraint
changes speaker labels only.

## 3. Multi-ASR lexical consensus

We use four public transcription views:

- **Qwen3-ASR-1.7B** with **Qwen3-ForcedAligner-0.6B** for multilingual text
  recognition and word-level timing;
- **FireRedASR2-AED** for a complementary Mandarin/dialect hypothesis with
  native timestamps;
- **FireRedASR2-LLM** for a language-model-heavy lexical hypothesis; and
- **MOSS-Transcribe-Diarize** for joint transcription, timing, and an
  independent speaker partition.

For a time-aligned lexical region, let $w_i^{(m)}$ be the token sequence from
recognizer $m$. A candidate replacement is accepted only when independent
recognizers agree after normalization and the replacement can be projected to
the frozen temporal support. This conservative consensus improves lexical
accuracy while avoiding a new segmentation search.

The MOSS hypothesis is retained as an important long-form view because its
speaker partition is produced jointly with transcription. The remaining ASR
systems provide independent evidence that is less coupled to that partition.

## 4. Acoustic speaker graph

### 4.1 Candidate activity tracks

We obtain complementary frame-level speaker activity from a full-conversation
streaming Sortformer and a public offline Sortformer. A label-free router
compares within-track acoustic coherence and selects the more reliable topology
for each recording. A speaker-count estimator controls the maximum active-role
capacity and triggers a CAMPPlus clustering fallback when required.

### 4.2 Multiscale speaker embeddings

For each frame center $t$, CAMPPlus and ERes2NetV2 embeddings are extracted
from windows of 0.75, 1.5, and 3.0 seconds. For encoder $q$, the feature is

$$
\mathbf{e}^{(q)}_t =
\left[\hat{\mathbf{e}}^{(q,0.75)}_t;
      \hat{\mathbf{e}}^{(q,1.5)}_t;
      \hat{\mathbf{e}}^{(q,3.0)}_t\right]\in\mathbb{R}^{576},
$$

where each 192-dimensional scale embedding is independently $L_2$-normalized.
Short windows preserve boundary sensitivity, whereas longer windows provide a
more stable identity estimate.

A residual metric adapter preserves the pretrained identity geometry and learns
only a gated correction:

$$
\mathbf{z}_t = \mathrm{norm}\left(
\bar{\mathbf{e}}_t + \sigma(g)f_\theta(\mathbf{e}_t)
\right).
$$

Pairwise same-speaker probabilities are computed from a calibrated cosine
similarity. These scores define edges between adjacent speech regions and are
combined with Sortformer activity to form a boundary-aware speaker graph.

### 4.3 Speech-region purity

An auxiliary dual-encoder purity network predicts silence, single-speaker
speech, or overlap for each frame. Both 576-dimensional encoder streams are
projected to 96 dimensions, processed by local temporal convolutions and a
two-layer bidirectional GRU, and classified into three states. Low-purity
regions are prevented from dominating speaker prototypes or boundary decisions.

## 5. Context-aware speaker Transformer

Local cosine similarity is unreliable for short turns, backchannels, and
overlap. We therefore encode the entire conversation before making speaker
decisions.

Let $\mathbf{a}_t,\mathbf{b}_t\in\mathbb{R}^{576}$ denote the ERes2NetV2 and
CAMPPlus multiscale features. Each stream is independently normalized and
projected:

$$
\mathbf{p}_t=\mathrm{GELU}(W_a\mathrm{LN}(\mathbf{a}_t)),\qquad
\mathbf{q}_t=\mathrm{GELU}(W_b\mathrm{LN}(\mathbf{b}_t)),
$$

with $\mathbf{p}_t,\mathbf{q}_t\in\mathbb{R}^{128}$. We concatenate the fused
feature with its first-order temporal difference,

$$
\mathbf{h}^{(0)}_t = W_f
\left[\mathbf{p}_t;\mathbf{q}_t;
\Delta\mathbf{p}_t;\Delta\mathbf{q}_t\right]\in\mathbb{R}^{256}.
$$

A depthwise temporal convolution with kernel size five supplies a local
residual. The resulting sequence is passed through a three-layer pre-norm
Transformer encoder with four attention heads, a 768-dimensional feed-forward
block, and dropout 0.15. The output has two heads:

- a normalized 192-dimensional contextual speaker embedding
  $\mathbf{c}_t$; and
- a scalar speaker-change logit $u_t$.

For sampled frame pairs $(i,j)$, the same-speaker logit is

$$
\ell_{ij}=\exp(s)\,\mathbf{c}_i^\top\mathbf{c}_j+b,
$$

where $s$ and $b$ are learned calibration parameters. Training minimizes

$$
\mathcal{L}=\mathrm{BCE}(\ell_{ij},y_{ij})
+\lambda_{\mathrm{chg}}\mathrm{BCE}(u_t,y_t^{\mathrm{chg}}).
$$

The contextual model is first trained for four epochs on 48 public
VoxConverse windows selected to match aggregate conversation structure, then
adapted for 14 epochs on fold-pure official development data. The deployment
model is refit on all 106 development conversations with the architecture and
epoch budget frozen beforehand.

## 6. Novel-speaker verification

A complementary partition can propose a role that is absent from the current
hypothesis. Accepting every proposal causes speaker fragmentation, while always
rejecting it causes deletions or speaker substitutions. We evaluate each
candidate with an independent WeSpeaker VoxCeleb ResNet34-LM encoder at 1, 2,
and 4-second scales.

For a candidate cluster, we compute 13 permutation-invariant statistics:

- support size, fragmentation, temporal span, and current speaker count;
- within-cluster cohesion at three temporal scales;
- similarity to the nearest existing speaker at three scales; and
- scale-wise distinctness, defined as cohesion minus nearest-speaker similarity.

A regularized logistic energy model estimates whether the candidate has
sufficient independent acoustic support. The fixed decision boundary is 0.5.
Because the features contain no absolute speaker index, the decision is
invariant to role permutation.

## 7. Speaker-assignment inference

The selected speaker tracks provide an initial partition. Multi-ASR consensus
supplies its lexical content. The acoustic graph and contextual embeddings then
score alternative role assignments for ambiguous regions. An assignment is
accepted only when its acoustic evidence, contextual metric, and purity guard
are compatible. A previously unseen role additionally has to pass the
WeSpeaker existence test.

This staged inference avoids a single monolithic score whose calibration would
mix unrelated failure modes. It also permits each neural component to be
trained with fold-pure supervision while the public ASR and speaker encoders
remain frozen.

## 8. Overlap-consistency constraint

Two simultaneously active regions cannot belong to the same physical speaker.
After neural inference, we inspect pairs assigned to the same role with at
least 0.20 seconds of overlap. A source role is restored only if:

1. the source partition assigns the two regions to distinct roles;
2. unchanged regions establish an unambiguous source-to-refined role mapping;
3. the proposed edit resolves the conflict; and
4. the edit creates no other same-speaker overlap.

This constraint never changes words, timestamps, ordering, or segment count.
It is deliberately limited to a physical inconsistency that is independent of
conversation topic and lexical content.

## 9. Optimization and deployment

All trainable speaker modules use fixed fold splits. The contextual model uses
Adam-style optimization with a learning rate of $3\times10^{-4}$, weight
decay $10^{-4}$, gradient clipping at 2.0, and bfloat16 computation. The
speaker-purity classifier uses class-weighted cross entropy; the contextual
metric uses balanced positive and negative frame pairs.

At deployment time, neural architectures, epoch budgets, probability
boundaries, and overlap thresholds are frozen before test inference. The test
set is not used for gradient updates, pseudo-label training, or per-session
parameter selection.

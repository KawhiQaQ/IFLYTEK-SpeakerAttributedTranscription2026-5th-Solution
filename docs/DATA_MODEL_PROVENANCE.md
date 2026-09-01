# Data and model provenance

## Official competition data

The official development set contains 106 labeled conversations. The test set
contains 394 unlabeled conversations and is used for inference only. Official
audio and annotations are not redistributed by this repository.

All supervised training, architecture selection, and error analysis use fixed
development folds. Competition test labels and pseudo-labels are never used,
and test audio is never included in a gradient update.

## External training data

The final contextual speaker model uses 48 short windows from the public
VoxConverse corpus. The selected windows contain approximately 42 seconds of
audio each. VoxConverse is used because it provides natural conversational
speaker changes and overlap rather than read speech.

Selection uses only aggregate, label-free conversation statistics such as
duration, speech occupancy, turn density, and overlap prevalence. It does not
use per-session test targets or pseudo speaker labels. The public RTTM labels
are used only after the external windows have been selected.

The contextual model is first pretrained on these public windows and then
adapted on the fold-pure official development split. External examples never
contain competition audio.

## Public pretrained models

| Component | Public model | Role in the system |
|---|---|---|
| Speech recognition | Qwen3-ASR-1.7B | multilingual transcript hypothesis |
| Forced alignment | Qwen3-ForcedAligner-0.6B | word-level temporal alignment |
| Speech recognition | FireRedASR2-AED | Mandarin/dialect transcript and timestamps |
| Speech recognition | FireRedASR2-LLM | language-model-heavy transcript hypothesis |
| Joint transcription and diarization | MOSS-Transcribe-Diarize | long-form transcript and speaker partition |
| Speaker diarization | NVIDIA offline and streaming Sortformer | candidate speaker activity tracks |
| Speaker representation | CAMPPlus | multiscale speaker identity and coherence |
| Speaker representation | ERes2NetV2 | complementary Chinese speaker embedding space |
| Speaker representation | WeSpeaker ResNet34-LM | independent novel-speaker verification |
| Voice activity detection | FSMN VAD | fallback speech activity detection |

Exact repository identifiers and local paths are listed in
`configs/public_models.yaml`. All public checkpoints remain subject to their
upstream licenses and model-card conditions.

## Test-distribution analysis

Unlabeled test audio is used only for aggregate distribution description and
final inference. No learned parameter is fitted on a test recording. The
aggregate profile is used to choose a public conversational corpus with similar
interaction structure; it is not used to assign or infer speaker identities.

## Licensing

- VoxConverse is released under CC BY 4.0; original media rights remain with
  their owners.
- Public pretrained models and source toolkits retain their upstream licenses.
- Competition data and trained checkpoints remain subject to the competition
  rules in addition to the repository's code license.

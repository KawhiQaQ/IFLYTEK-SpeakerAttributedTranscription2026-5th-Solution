# Third-party notices

This project composes public datasets, pretrained models, and toolkits. They are not relicensed by the repository's Apache-2.0 license.

## Source toolkits

| Component | Upstream | Frozen source revision | License |
|---|---|---|---|
| MOSS-Transcribe-Diarize | https://github.com/OpenMOSS/MOSS-Transcribe-Diarize | `cb765f2b0fe6f7a298aa2002e2281ae693d1f3c3` with a documented import fallback; executed with Transformers 5.16.1 | Apache-2.0 |
| FireRedASR2S | https://github.com/FireRedTeam/FireRedASR2S | `4e7d9aaf4482a47cec1724807026b9b151926eb5` | Apache-2.0 |
| 3D-Speaker | https://github.com/modelscope/3D-Speaker | speaker embedding and fallback diarization toolkit | Apache-2.0 |
| DiariZen / vendored pyannote.audio | https://github.com/BUTSpeechFIT/DiariZen | `844f5555b0a98acd0931511fc641a8c5b8ba92c7` | MIT |
| MeetEval | https://github.com/fgnt/meeteval | installed as `meeteval==0.4.3` | MIT |

## Data

- The official competition dataset is governed by the competition terms and is not redistributed.
- VoxConverse v0.3 is released under CC BY 4.0; original video copyright remains with its owners. The exact fixed-window provenance is recorded in `manifests/voxconverse_fixed_48.json`.

## Models

Model IDs, expected files, and recorded hashes are in `configs/public_models.yaml`. Consult each upstream model card before redistribution or commercial use.

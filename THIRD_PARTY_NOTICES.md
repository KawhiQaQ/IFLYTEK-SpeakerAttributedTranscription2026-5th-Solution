# Third-party notices

This project composes public datasets, pretrained models, and toolkits. They are not relicensed by the repository's Apache-2.0 license.

## Source toolkits

| Component | Upstream | Frozen source revision | License |
|---|---|---|---|
| MOSS-Transcribe-Diarize | https://github.com/OpenMOSS/MOSS-Transcribe-Diarize | `cb765f2b0fe6f7a298aa2002e2281ae693d1f3c3` | Apache-2.0 |
| FireRedASR2S | https://github.com/FireRedTeam/FireRedASR2S | `4e7d9aaf4482a47cec1724807026b9b151926eb5` | Apache-2.0 |
| WeSpeaker | https://github.com/wenet-e2e/wespeaker | `dfa741957e5c11f477623b6e583d67d0af25ee88` | Apache-2.0 |
| 3D-Speaker | https://github.com/modelscope/3D-Speaker | speaker embedding and fallback diarization toolkit | Apache-2.0 |
| MeetEval | https://github.com/fgnt/meeteval | installed as `meeteval==0.4.3` | MIT |

## Data

- The official competition dataset is governed by the competition terms and is not redistributed.
- VoxConverse v0.3 is released under CC BY 4.0; original video copyright remains with its owners. The exact selected-window provenance is recorded in `data/external/voxconverse_test_matched/audit.json` after restoring the runtime bundle.

## Models

Model IDs, expected files, and recorded hashes are in `configs/public_models.yaml`. Consult each upstream model card before redistribution or commercial use.

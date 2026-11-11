# Sustained spoken tabletop control

Both examples run the locked spatial model at seed 1411. They were selected by fixed mission index before their outcomes were observed. Each starts from an independently seeded scene. Six genuine human utterances arrive during continuous contact simulation. The view is sampled once per 300 ms macro, the physics clock runs at 10 Hz, and the file carries the original audio on its simulation clock. The controller receives each cached decoder result only after that recording finishes. The overlaid named stations are part of the public map.

![The running tabletop policy at 25 seconds in the first mission](mission-00-preview.png)

- [mission-00 · 61.2 seconds](mission-00.mp4) · [Native positions, applied controls and retained goals](mission-00.npz)
- [mission-08 · 61.2 seconds](mission-08.mp4) · [Native positions, applied controls and retained goals](mission-08.npz)

Across 96 instructed phases the retained-goal spatial and dense policies each completed 54 selective placements. Geometric feedback completed 47. The paired spatial gain over geometry was +7.3 [+1.0, +13.5] percentage points.

[Every mission and cue outcome](mission-outcomes.json) · [48 full native held trajectories](native-traces.npz) · [Independent replay audit](../metrics/trajectory-replay.json)

The human recordings are from Martin Cooke, Jon Barker, Stuart Cunningham and Xu Shao's *The Grid Audio-Visual Speech Corpus*, CC BY 4.0. Audio was aligned to the recording clock and placed at cue onset. The control scene is generated in Google's open Language-Table PyBullet environment. [Original speech corpus](https://doi.org/10.5281/zenodo.3625687) · [Speech license](https://creativecommons.org/licenses/by/4.0/) · [Simulation source](https://github.com/google-research/language-table)

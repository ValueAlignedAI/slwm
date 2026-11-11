# Recorded speech and continuous device forecasts

These four fresh recordings supply a 603-second operational replay. Each stream has a one-second observation clock plus an update at each completed prospective directive. The choice of recordings used source duration and the count of completed directives, prior to scoring predictions. Foreground actions and later images play no part in deciding when the model runs. A second file contains the ground-truth timing used for post-run evaluation. There is one current model decision per line. [Stream inventory, source recordings and audio timestamps](streams.json) · [Single-clock execution](../metrics/single-clock.json)

| Unseen family | Duration | Update clocks | Actual and earlier spoken instruction | Associative decisions |
| --- | ---: | ---: | --- | --- |
| Coffee machine | 180 s | 191 | [Actual](coffee-true.wav) · [Earlier](coffee-foil.wav) | [Read decisions](coffee-decisions.jsonl) |
| Computer | 129 s | 137 | [Actual](computer-true.wav) · [Earlier](computer-foil.wav) | [Read decisions](computer-decisions.jsonl) |
| Camera | 180 s | 192 | [Actual](camera-true.wav) · [Earlier](camera-foil.wav) | [Read decisions](camera-decisions.jsonl) |
| Printer | 114 s | 122 | [Actual](printer-true.wav) · [Earlier](printer-foil.wav) | [Read decisions](printer-decisions.jsonl) |

At 135.5 seconds in the camera recording, the instructor says “The next step is to turn on the GoPro.” The earlier instruction says “The next step is to change the micro sd card.” The released associative model kept the actual cue internal and issued three “grab” proposals under the earlier cue. The example was chosen by source hash. It illustrates sensitivity to speech and the limits of that sensitivity for selecting a useful action. [Verified raw-waveform outputs](../metrics/raw-audio.json)

The source microphone stream is publicly pitch-shifted. Audio clips, images and derived signals come from the HoloAssist dataset of Wang and colleagues. [HoloAssist](https://holoassist.github.io/) · [Data agreement](HoloAssist-LICENSE.txt)

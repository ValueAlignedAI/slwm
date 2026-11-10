# G5 joint predictive state

Registered November 10 before looking at G5 development or test outcomes.

G4 trained conditional generators on real paired views. Dense attention benefited from temporal negatives on video-to-audio prediction. The signal core failed to replicate the advantage, generated GRID speech had 100% word error, and robot state forecasts lost to constant velocity. G3 found an advantage over a matched GPT on spoken digits that a raw-feature classifier exceeded. The new question is whether a shared state can predict **which outcome changes when an action changes** and **which sound accompanies a visible contact**. These claims are separable. [G4 report](../../t4/crossmodal-state-tpu-20261109/report.md)

## Comparisons

Train a learned-retention hybrid with periodic dense attention and an all-dense GPT-2-style causal Transformer from scratch. Use identical depth, width, physical edges, shared 128-dimensional state, heads, initial parameter tensors, source exposure, optimizer, batch, and updates. Report parameter counts, throughput, wall time, exposure and unique recordings. Equal example and parameter budgets may consume different device time. The experiment does not use OpenAI's pretrained GPT-2 weights. Grouped or 2-of-4 sparse extensions require a separate registration.

The shared state can see multiple synchronized modalities during pretraining. A target modality is removed whenever its cross-modal decoder is scored. Sample audio, video, numeric sensors and byte text only when actually present. A text description at the start of an example is privileged conditioning. Action plans are known control inputs. Withhold future sensors, frames and outcome labels from the causal dynamics predictor. Center the Push-T endpoint on the last observed block pose and share the same observed history across three different prospective actions.

## Primary readouts

* On Push-T, measure mean Euclidean block-center displacement error at 16 actions, in arena pixels. In matched three-plan groups also report the error on *differences* between predicted plan outcomes. Compare with persistence, constant velocity and the dense control. Bootstrap whole reset seeds.
* On real Greatest Hits recordings, measure held-out video-only contact timing by average precision on annotated positive contact bins. Use annotation fields with an identified material, exclude the release's all-None entries from positive labels, and report prevalence. Compare within-recording shifted video, erased video, and the dense control. Bootstrap whole recording dates. Windows are chronological and nonoverlapping within recordings. [Original dataset](https://andrewowens.com/vis/)
* A joint architecture win requires at least 5% relative improvement on both primary endpoints against dense in both initialization seeds, with paired 95% intervals excluding zero, and positive control discrimination. Report task-specific findings even when this gate fails. Simulated planning evidence requires a separate closed-loop evaluation. Speech intelligibility requires decoding and word error on independently held speakers. A lower regression loss alone supports neither conclusion.

## Scale and test separation

The fixed training mixture and endpoints are in the study configuration. Development is for throughput, stopping and deciding promotion. Use two paired seeds in published comparisons. Reserve test roles until both models and both seeds at the promoted scale have complete, hashed weights. Use the pilot to detect mechanical failures and evaluate validation feasibility. Promote only if both architectures train finitely and their development comparisons against the erased or zero-displacement controls show usable signal on both primary tasks. If the pilot is inconclusive under this rule, report the limitation and preserve the test set. An anchor is targeted at 8,192 updates of 256 examples, or 134,217,728 sequence positions per arm and seed. Running the entire ladder depends on measured hardware time and the aggregate credit limit.

Seven existing sources include paired speech, images and captions, environmental sounds, DROID robot sensors and actions, natural meeting audio/video, and text. Greatest Hits adds impacts and the pinned Push-T simulator adds controlled branches. Report actual usable counts and exact splits at each stage. Acquisition of LibriSpeech train-clean-360 is a separate expansion. It cannot count toward training until decoded, split and ingested. [LibriSpeech](https://openslr.org/12/) · [Push-T simulator](https://github.com/huggingface/gym-pusht)

The accessible Google target is single-chip v6e. Use separate assignments for each architecture when officially available, and stop both at the aggregate project cap. Keep large data on Fedora SSD or Drive and models on checkpoint HDDs or Drive. Every promoted model needs a restorable optimizer and a verified inference archive. Publish only aggregate, reproducible evidence to Git.

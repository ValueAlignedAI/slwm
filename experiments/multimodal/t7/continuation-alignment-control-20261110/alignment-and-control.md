# What the speech and decision controls establish

## Speech timing survives better than speech content

Eight fixed 2,048-update heads finished. Both architectures received identical masked inputs and corpus proportions. On eight unseen CREMA-D actors, the hybrid chose the correct audiovisual pairing among the true pairing, an eight-tick delay and another utterance by the same actor in 61.7% and 66.7% of trials. Dense GPT-2 scored 55.8% and 54.2%. Chance is 33.3%. The ordering reversed on real sound events at the held Tokyo site. The hybrid scored 48.1% and 50.3%, and the dense model scored 53.0% and 51.0%. The acoustic codebook contains 64 entries. Even on acted speech, correct video recovered only 12.1–12.7% of unit labels. [Held fields and checkpoints](metrics/alignment.json)

| Frozen recognizer input | Acted speech WER | Audiobook WER |
|---|---:|---:|
| Original waveform | 12.5% | 14.8% |
| Fine acoustic features through Vocos | 15.3% | 13.6% |
| Four-frame-averaged acoustic features through Vocos | 44.6% | 34.2% |
| Correct acoustic units and pooled voice | 84.7% | 99.2% |
| Correct acoustic units and separate same-speaker enrollment | 82.9% | 100.6% |

The fixed waveform sample contains 64 utterances per corpus. The scorer compares the portion of the annotated transcript enclosed by the clip with a greedy Whisper-tiny.en transcription. Clipping, recognition error and partial boundary words affect all conditions. The fine-feature control sets an empirical floor for this measurement. Both unit prediction and the unit-to-waveform path lose content. Giving the renderer the correct units still leaves most intended words unrecoverable. A listener-preference objective would have a weak content signal at this point. The acoustic teacher was pretrained on LibriSpeech 960h, which includes the 360h source from which SLWM's fresh speaker splits were drawn. [Per-condition word counts, intervals and model hashes](metrics/alignment.json)

## A good supervised ranker can miss the terminal goal

On the four-cycle development task, the train-only physical prior predicted held endpoints to 0.45 cm and ranked the best action 95.1% of the time when supplied the true goal. With this same prior, the best supervised adapters reached 0.15–0.62 cm regret. A raw-observation recurrent head reached 0.05–0.06 cm. The matched worlds recovered the early cue, but this task also has a simpler sufficient representation. Before introducing the common physical prior, frozen heads failed the development criterion. [Supervised experiment](supervised-followup.md)

Each of four residual actors then trained on 11,520 real on-policy simulator decisions. At 24 consecutive decisions, final goal success for the hybrid went from 1.6% to 4.7% on the first seed and from 5.5% to 6.3% on the second. Dense went from 7.0% to 3.9% and from 2.3% to 0.8%. Every paired interval for RL versus its supervised parent included zero. In the 192-decision diagnostic, both hybrid actors entered the goal at some point in 81% of episodes but ended there in none of the 16 episodes per seed. Completing a long stream and maintaining a goal are distinct measured outcomes. [Closed-loop success, controls and uncertainty](metrics/closed-loop.json)

With exact physics, one-step greedy planning ended inside the goal in 7.0% of the original episodes. Two-step lookahead reached 48.4%. A second group of 128 episodes gave 12.5% and 44.5%. Both oracle comparisons use the true goal, and the two-step oracle can inspect the next exogenous proposal library. These post hoc measurements motivated a separate, prospectively locked learned-planner experiment on new starting states. [Planner protocol](planning-followup.md)

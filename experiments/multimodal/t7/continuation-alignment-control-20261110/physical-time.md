# Physical timing on independent people and rooms

The confirmation set adds 336 clips from 14 speakers outside every earlier G7 CREMA-D split. It also adds 1,250 windows from 28 synchronized recordings in two previously unused rooms of the official STARSS23 Tokyo partition. Each head learned from the same G7 training people and Finnish rooms. Identical crop duration, source labels, position and validity masks were supplied to the four choices. Removing sensor values made every encoded camera alternative identical. [Prospective design](raw-time-followup.md) · [Input hashes and results](metrics/physical-timing.json)

| Four-way selection accuracy | New speakers, seed 711 / 712 | New Tokyo rooms, seed 711 / 712 |
|---|---:|---:|
| Original hybrid head on physically shifted input | 23.2% / 26.8% | 14.4% / 13.4% |
| Original dense head on physically shifted input | 24.7% / 27.4% | 13.8% / 17.5% |
| Hybrid field, head trained with both signs | 43.8% / 42.9% | 14.7% / 16.2% |
| Dense GPT-2 field, head trained with both signs | 42.6% / 39.9% | 15.7% / 15.7% |
| Fixed raw sensor projection, same trained head | 68.8% / 65.2% | 21.5% / 20.2% |
| Raw spectral flux and pixel motion | 36.9% | 25.7% |
| Uniform choice | 25.0% | 25.0% |

The matched four alternatives are correctly timed video, an older camera view, a newer view and different footage from the same speaker or room. The original local three-way score lost most of its apparent advantage when the physical clocks were corrected. A trained readout using the frozen shared field recovered some timing information about acted speech. Training the same readout directly on the raw sensors recovered substantially more. The small held-room sample supports only a statement about these two rooms. [Directional checks and paired actor/room intervals](metrics/physical-timing.json)

A second intervention varied the direction of the time negatives while keeping the head, number of negatives and number of updates fixed. On the 14 speakers, changing to both signs raised hybrid four-way accuracy by 3.6 and 5.1 percentage points. The first seed's actor-bootstrap interval includes zero. Positive scores relative to the earlier combined head also reflect a change to the dedicated timing objective. [All matched variants](metrics/physical-timing.json)

## Training shares

The continuation schedule has 2,097,152 examples per arm. Audio was present in 50.0% of rows, video in 75.0%, numeric state in 31.25%, and text in 51.7%. Audio conditioned 25.0%, video 42.9%, numeric state 21.1%, and text 30.6%. Rows can condition on two sensors. Text-only language modeling accounts for 9.375% of all rows. The remaining text exposure includes paired captions, utterances and other annotated observations. There are 458,752 action-conditioned future rows and 229,303 vision-conditioned event-target rows per arm. These are presentation counts; loss magnitude and gradient allocation also depend on the objective. [Full schedule](metrics/data.json)

The fresh arm changed exactly 752,149 rows (35.87%). It drew 58,994 independent audiobook examples, 98,304 document windows from distinct source documents, 16,384 simulator starts, 3,761 room-event windows and 1,650 complete acted-speech windows. Recorded training inventory and effective sampling exposure differ because the continuation preserved the original validity masks. [Source inventory, unique visits and repeats](metrics/data.json)

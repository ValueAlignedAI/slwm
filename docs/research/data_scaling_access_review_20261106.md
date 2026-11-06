# Paired-data scale and access review, 2026-11-06

This is an acquisition plan, recorded before the B1/P1 final scores; no new
corpus or fitted candidate is added to those comparisons.

The practical first expansion is more independent AMI groups and sessions.
The [official AMI page](https://groups.inf.ed.ac.uk/ami/corpus/) describes a
100-hour multimodal corpus with synchronized microphones/cameras, transcripts
and CC BY4.0 signal/transcription releases. Current B1 uses only about2.05 hours
of distinct common training-meeting time, despite8.23 hours of parallel person
tracks. The remaining corpus is a candidate pool, not verified downloaded
support: the present source audit already encountered403s for one meeting and
measured video endings shorter than AVI headers.

Before expanding, freeze participant-connected meeting groups across all prior
training, selection, calibration and test exposures. Grouping only utterances
or session suffixes can repeat the same people. Validate every selected camera's
decoded clock and common audio/video support; record missing sources before
fitting. Budget active own-speaker windows deliberately while retaining quiet
and overlapping-speech evaluation. Train-only normalization, real phone
alignments and a measured temporal mouth edge should accompany the data study.
Keep a fixed small model and paired Transformer as capacity controls before
attributing improvement to a125M joint processor. This remains meeting/face
data, rather than general scene or action-transition supervision.

[MuAViC](https://github.com/facebookresearch/muavic) reports1,200 hours in nine
languages and436 English training hours. Its
[actual loader](https://github.com/facebookresearch/muavic/blob/main/get_data.py)
requires a separately supplied LRS3 corpus for English; the paper's larger
number is not an independently available English download. The non-English
[pipeline](https://github.com/facebookresearch/muavic/blob/main/mtedx_utils.py)
fetches OpenSLR audio/annotations and reconstructs video from YouTube, so a
bounded availability/timing pilot is necessary. The repository states CC-BY-NC4.0
and is archived. No MuAViC media or pretrained model is used in B1.

The [official VoxCeleb2 download page](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
explicitly says that URLs/timestamps, audio, video and identifying metadata are
no longer available there. A historical dataset citation alone cannot justify
an immediately runnable acquisition budget. The Oxford LRS3 page also returned
404 in this check. No unofficial mirror was adopted as if it were a verified
replacement.

For action-conditioned work, the separately retained DROID preview establishes
two actual episodes and typed camera/proprioception/command fields. Expand to
episode/environment/session-disjoint source groups, verify cadence and proper
rotation metrics, then register matched no-action, shuffled-action, direct-policy
and Transformer controls. A larger processor is a subsequent capacity question;
the current two debug episodes are insufficient training data. See the separate
action-conditioned next-stage contract and source probe for the bounded evidence.

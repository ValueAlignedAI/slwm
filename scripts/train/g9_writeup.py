"""Write the G9 report from all frozen rows, controls and custody receipts."""
import argparse
import json
from statistics import median
from pathlib import Path

STUDY=Path("experiments/multimodal/t9/concurrent-memory-20261110")


def main(repo):
    folder=repo/STUDY;metrics=folder/"metrics";doc=json.loads((metrics/"results.json").read_text())
    release=json.loads((metrics/"release.json").read_text());card=json.loads((metrics/"data-card.json").read_text())
    if doc["status"]!="complete" or not release["complete"]:raise ValueError("A complete G9 release is required")
    rows=doc["rows"];pairs=doc["comparisons"];other=doc["followups"]
    def id(size,core,field,seed,act="gelu",source="fresh",role="test",step=None):
        return f"{size}-{core}-z{field}-{act}-s{seed}-{source}/{step or (384 if size=='pilot' else 4096)}/{role}"
    def series(fn,scale=1,digits=2):
        return " / ".join(f"{fn(seed)*scale:.{digits}f}" for seed in (911,912))
    def score(size,core,field,metric,scale=1,digits=2,role="test"):
        return series(lambda s:rows[id(size,core,field,s,role=role)]["mean"][metric],scale,digits)
    def interval(obj,scale=100,digits=1):
        return f"{obj['mean']*scale:+.{digits}f} [{obj['ci95'][0]*scale:+.{digits}f}, {obj['ci95'][1]*scale:+.{digits}f}]"
    def gaps(family,size,core,field,metric,act="gelu"):
        return " / ".join(interval(pairs[family][id(size,core,field,s,act=act)][metric]) for s in (911,912))
    def horizon(core,mode,field=512):
        return series(lambda s:other["history_window"][f"confirm-{core}-z{field}-gelu-s{s}-fresh"][mode]["occupancy"],100,1)
    table=[]
    for label,size in (("13M, 384 updates","pilot"),("13M, 4,096 updates","exposure"),("117M, 4,096 updates","confirm")):
        for field in (128,512):
            table.append(f"| {label} | {field} | {score(size,'hybrid',field,'live/carry/occupancy',100,1)} | {score(size,'gpt2',field,'live/carry/occupancy',100,1)} |")
    density=card["runs"]["911-4096"]
    real_table=[]
    for key,title in (("a","Audio log-band MSE"),("v","RGB MSE"),("n","Normalized physical-channel MSE"),("w","Event-word binary cross-entropy")):
        real_table.append("| "+title+" | "+" | ".join(score("confirm",core,512,f"room/{key}/{key}/{mode}",1,4) for core,mode in
            (("hybrid","aligned"),("hybrid","shifted"),("gpt2","aligned"),("gpt2","shifted")))+" | "+
            f"{rows[id('confirm','gpt2',512,911)]['mean'][f'prior/{key}/{key}/aligned']:.4f} |")
    robot=" / ".join(f"{other['uncapped_contact'][f'confirm-{c}-z512-gelu-s{s}-fresh']['mean_rmse_newtons']['frozen']:.2f}" for c in ("hybrid","gpt2") for s in (911,912))
    rank=lambda c,w:series(lambda s:rows[id("confirm",c,w,s)]["real_field_rank"]["participation_ratio"],1,2)
    cache=lambda c:other["history_window"][f"confirm-{c}-z512-gelu-s911-fresh"]["carry"]["cache_bytes_per_world"]
    q=other["backend_check"];max_backend=max(abs(v["tpu_minus_cuda"]["mean"]) for v in q.values())*100
    wide_adv=" / ".join(interval(other["history_architectures"][f"z512-s{s}"]["block32"]) for s in (911,912))
    gate=0
    for c in ("hybrid","gpt2"):
        for s in (911,912):
            if rows[id("pilot",c,512,s,"snake",role="dev")]["mean"]["live/carry/occupancy"]>rows[id("pilot",c,512,s,role="dev")]["mean"]["live/carry/occupancy"]:gate+=1
    assert gate<4
    runtime={}
    for kind in ("exposure","confirm"):
        timing=[v for name,v in doc["training_cost"].items() if name.startswith(kind+"-")]
        runtime[kind]=tuple(median(v[k] for v in timing) for k in ("end_to_end_train_seconds","median_warm_updates_per_second"))
    dose=other["independent_episode_dose"]
    def dose_line(core,name):
        return series(lambda seed:dose[f"{core}-s{seed}"]["carry"][name],100,1)
    text=f"""# G9 shared state, training exposure and sustained decisions

On held 51.2-second missions, the 13M hybrid learned to use a past calibration tap, reach a requested station and hold position through changing goals. At 4,096 updates its 128-coordinate field reached **{score('exposure','hybrid',128,'live/carry/occupancy',100,1)}%** goal occupancy across the two training seeds. The equal-parameter dense GPT-2-style model reached **{score('exposure','gpt2',128,'live/carry/occupancy',100,1)}%**. A physical controller that knew how to decode the same sensors reached 96.24%. Holding forever reached 6.54%. The learned advantage is concentrated in this persistent-control setting. Recorded multimodal transfer gives a different ordering.

![Matched exposure, history and real-sensor scores](g9-comparisons.png)

Open and filled points show seeds 911 and 912. The dotted line in the first panel marks the sensor-based controller. Positive skill in the real-sensor panel means lower loss than the training prior.

## Exposure and width

All figures below are percentages of time at the active goal. Each cell shows independent training seeds 911 and 912 on the same 128 held missions. The held simulator uses previously unused mission seeds. Layouts and actuator signs come from the same families as training, and the intermittent motor-stall rule is held out. Each core issued its own controls for 512 successive ticks. Matched architectures share initial common matrices, training examples, masking draws and parameter counts within a field width.

| Trunk and training | Emitted coordinates | SLWM | Dense GPT-2 |
| --- | ---: | ---: | ---: |
{chr(10).join(table)}

At equal 4,096-update exposure, the large-minus-small SLWM change for the 128-coordinate field was {gaps('large_minus_small_equal_exposure','confirm','hybrid',128,'live/carry/occupancy')} percentage points. Moving the large SLWM from 128 to 512 coordinates changed occupancy by {gaps('wide_minus_narrow','confirm','hybrid',512,'live/carry/occupancy')} points. The full-size 128-coordinate SLWM-minus-dense differences were {gaps('hybrid_minus_dense','confirm','hybrid',128,'live/carry/occupancy')}. Intervals resample entire missions 4,000 times within each training seed. Exposure and optimization explain the early failure of the small hybrid on this task. The small and large fresh runs consumed about {doc['training_cost']['exposure-hybrid-z128-gelu-s911-fresh']['sampled_positions_per_parameter']:.2f} and {doc['training_cost']['confirm-hybrid-z128-gelu-s911-fresh']['sampled_positions_per_parameter']:.2f} sampled clock positions per parameter.

## Independent data at equal training exposure

| 13M wide-field model, 4,096 updates | 393,216 independent starts | Replay of 2,048 starts |
| --- | ---: | ---: |
| SLWM | {dose_line("hybrid","fresh")}% | {dose_line("hybrid","replay")}% |
| Dense GPT-2 | {dose_line("gpt2","fresh")}% | {dose_line("gpt2","replay")}% |

At 384 updates, 36,864 fresh simulator episodes versus replay of 2,048 episodes gave a hybrid occupancy change of {gaps('fresh_minus_replay','pilot','hybrid',512,'live/carry/occupancy')} points. The short screen gave no replicated benefit from the larger episode pool at fixed update count. At 4,096 updates and 393,216 fresh versus 2,048 replayed starts, the independent-cohort fresh-minus-replay occupancy differences were {" / ".join(interval(other["independent_episode_dose"][f"hybrid-s{s}"]["carry"]["fresh_minus_replay"]) for s in (911,912))} points for SLWM and {" / ".join(interval(other["independent_episode_dose"][f"gpt2-s{s}"]["carry"]["fresh_minus_replay"]) for s in (911,912))} for dense GPT-2. The real recordings, masks, number of updates and starting parameters were identical within each comparison. [Independent-dose registration](data-dose.md) The table uses a new cohort of 128 missions. The long-dose test was registered after observing the short-screen floor.

## Why history matters

In the primary large 128-coordinate hybrid, resetting memory on every observation yielded {score('confirm','hybrid',128,'live/reset/occupancy',100,1)}% occupancy. Reversing the sign of the previously issued tap yielded {score('confirm','hybrid',128,'live/wrong/occupancy',100,1)}%. On paired histories with identical audio, video, physical and word streams, the issued tap was the only distinguishing input. SLWM selected the correct final action in {score('confirm','hybrid',128,'history/matched',100,1)}% of cases. Dense GPT-2 reached {score('confirm','gpt2',128,'history/matched',100,1)}%.

A separate cohort changed only inference. The large wide-field hybrid reached {horizon('hybrid','carry')}% with the native 64-step attention cache and {horizon('hybrid','full')}% with a full 512-step cache. Dense GPT-2 reached {horizon('gpt2','carry')}% and {horizon('gpt2','full')}%. Resetting **both** architectures every 32 ticks, matching the training sequence length, gave {horizon('hybrid','block32')}% for SLWM and {horizon('gpt2','block32')}% for dense GPT-2. The remaining paired gap was {wide_adv} points. Context-length mismatch explains a substantial part of the native dense deficit. Native per-world layer caches occupied {cache('hybrid'):,} bytes for SLWM and {cache('gpt2'):,} bytes for dense GPT-2. These counts cover recurrent states, keys, values and validity masks. [Individual memory interventions](memory-and-adaptation.md)

## What the recorded modalities carry

Every training window has a common 10 Hz clock. Each step can contain 64 spectral measurements, a 16×16 RGB image, 12 physical channels and eight small-vocabulary word slots. Before masking, all streams are available. In seed 911, {100*density['all_four_fraction']:.2f}% of training windows had all four visible. Three quarters of rows are simulated and one quarter come from 50 STARSS23 recordings spanning seven training rooms and 2.87 hours. At 4,096 steps there are 393,216 independent simulated starts and 16,777,216 sampled clock positions. The real draws visit {density['real_distinct_window_starts']:,} distinct overlapping window starts and {density['real_distinct_clock_bins']:,} distinct recorded ticks. The same room, static board image or repeated goal word can contribute many measurements while adding little independent information.

We withheld the target throughout each window on 30 Sony recordings from two held rooms. The aligned and half-recording-shifted conditions use identical source masks. Lower loss is better. The prior uses training statistics. The word targets name annotated events.

| Target | SLWM aligned | SLWM shifted | Dense aligned | Dense shifted | Prior |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(real_table)}

Both architectures improved on the training prior for tracked geometry. Dense GPT-2 also improved on the audio prior in both seeds. The prior had lower RGB and event-word loss than either wide model, even though correct timing improved all four target losses over the shifted-source control.

The large hybrid's held-room covariance participation ratios were {rank('hybrid',128)} for 128 coordinates and {rank('hybrid',512)} for 512. The corresponding dense values were {rank('gpt2',128)} and {rank('gpt2',512)}. These describe variance on the held room panel. Testing whether events occupy common coordinates across sources requires a separate measurement. The 40-second audio-to-RGB, video-to-audio and event-word-to-both examples preserve that distinction. [View and listen](examples/README.md)

A separate within-recording retrieval test used all 28 available recordings in two further Sony rooms. Each single-source query selected among eight time windows of the same recording, with chance 12.5%. With a 512-coordinate field, direct audio-to-vision retrieval reached {series(lambda s:other["source_retrieval"][f"confirm-hybrid-z512-gelu-s{s}-fresh"]["audio_to_vision"]["direct"],100,1)}% for SLWM and {series(lambda s:other["source_retrieval"][f"confirm-gpt2-z512-gelu-s{s}-fresh"]["audio_to_vision"]["direct"],100,1)}% for dense GPT-2. After an orthogonal alignment fitted only to the TAU training recordings, those scores were {series(lambda s:other["source_retrieval"][f"confirm-hybrid-z512-gelu-s{s}-fresh"]["audio_to_vision"]["train_aligned"],100,1)}% and {series(lambda s:other["source_retrieval"][f"confirm-gpt2-z512-gelu-s{s}-fresh"]["audio_to_vision"]["train_aligned"],100,1)}%. This measures temporal correspondence beyond room identity. Event words and tracked-event counts share an annotation process, which can also produce an association. [Twelve directional comparisons and raw activity controls](metrics/invariance) · [Retrieval protocol](source-invariance.md)

The ROMA transfer panel froze every backbone and fitted a common ridge head on six calibration scenes. Tests used 36 trials from eighteen other scenes with timestamps joining contact audio, camera images, action names and force. The target was next-tick force while the physical input was withheld. On uncapped original targets, the two large wide hybrids and the two dense controls gave **{robot} N**, in that order. The training-mean reference gave 6.95 N, raw non-force sensors 7.10 N, and a reference allowed to see the current force 3.26 N. The mean baseline had lower error than all four wide parents on the original force range. The registered ±20 N range clipped at least one component in 18.4% of held ticks. [Original-range sensitivity](metrics/uncapped) · [Dataset and identifiers](metrics/inputs.json)

## Negative findings and next experiments

- Fixed-frequency Snake passed {gate} of four wide-field development comparisons. The promotion rule required all four. A large periodic arm was therefore unnecessary under this protocol. This result applies to feed-forward Snake on these quantized input features.
- Fitting a 66,567-parameter decision head on 16,384 parent-visited states reduced fresh-cohort occupancy in all four wide large parents. In the two hybrids, 14,705 and 14,364 training labels requested hold, compared with 975 for random trajectories. Visit distribution changed materially. The experiment does not isolate label imbalance as the cause. [Head weights and controls](metrics/followups/heads)
- The four-layer gated recurrent control was tested at 384 updates. Its wide-field occupancy was {score('pilot','gated',512,'live/carry/occupancy',100,1)}%. Establish its equal-exposure performance and compare stronger recurrent baselines before generalizing the architecture ranking.

The next decision experiment is to train both architectures on matched 32–512-step context distributions at equal exposure, then test unpredictable goal changes and actuator recalibration. The next data investment is additional independently recorded, synchronized natural instructions, contacts and outcomes across sites. Test an alignment target that distinguishes the correct event from same-room, wrong-time evidence. Require better direct retrieval and conditional prediction on new rooms while preserving the learned action memory. Keep whole sites and novel dynamics held, compare wrong-time and wrong-instruction inputs, and require performance above a physical and persistence reference. The action result supports sustained-memory research. The real cross-modal results justify more diverse grounded data before increasing the shared width again.

## Artifacts and compute

All 42 fits completed and {doc['two_disk_checkpoints']} inference/optimizer generations were hash-verified on both Fedora checkpoint volumes. Native TPU and CUDA inference independently replayed the sixteen final equal-exposure and large models from identical weight hashes. The largest difference in mean goal occupancy between the backends was {max_backend:.2f} percentage points. Each paired architecture comparison uses a common backend. A single changed action can change subsequent observations.

On the same TPU at 4,096 updates, median warm throughput was {runtime["exposure"][1]:.1f} updates/s for the small trunks and {runtime["confirm"][1]:.1f} for the large trunks, at 128 sequences of 32 ticks. Median elapsed training time, including compilation and checkpoint work, was {runtime["exposure"][0]:.1f} and {runtime["confirm"][0]:.1f} seconds per fit, respectively.

G9 used {release['owned_units']:.2f} project Colab units on a named v5e-1. The four-layer screen trained on the RTX 3090. Cumulative owned usage for G7–G9 was {release['cumulative_owned_units']:.2f} of the 100-unit allowance. The G9 runtime was released. At release, the Google Drive mirror was backing off from its project query limit, with {release["drive_at_release"].get("verified_files",0)} G9 files verified there. The public packages give both complete small backbones and the exact inference CLI. The private optimizer stages and unquantized weights are indexed by tag, step and SHA-256. [Download and run](inference/README.md) · [Artifact inventory](metrics/artifacts.json) · [Paired observations](metrics/paired-observations.npz) · [Registered protocol](protocol.md) · [Full machine-readable results](metrics/results.json)
"""
    (folder/"report.md").write_text(text)
    path=repo/"README.md";s=path.read_text()
    start=s.index("### G9 concurrent sensing and sustained decisions");end=s.index("### G8 acoustic coding",start)
    section=f"""### G9 concurrent sensing and sustained decisions

With 4,096 updates, a 13M SLWM and a 128-coordinate shared output spent {score('exposure','hybrid',128,'live/carry/occupancy',100,1)}% of the time at the requested goal across two seeds in 51.2-second held missions. The matched dense GPT-2 reached {score('exposure','gpt2',128,'live/carry/occupancy',100,1)}%. Larger trunks and 512-coordinate outputs gave no consistent gain. Resetting both large wide-field models at their 32-step training length greatly narrowed the dense gap. On real room recordings and unseen robot contacts, the architecture ranking depended on the target and neither large wide model consistently beat the simple force reference. [Full matched study](experiments/multimodal/t9/concurrent-memory-20261110/report.md) · [Two complete 5.5 MB models](experiments/multimodal/t9/concurrent-memory-20261110/inference/README.md) · [51-second decisions and 40-second generation](experiments/multimodal/t9/concurrent-memory-20261110/examples/README.md)

"""
    path.write_text(s[:start]+section+s[end:])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("repo",type=Path);main(p.parse_args().repo)

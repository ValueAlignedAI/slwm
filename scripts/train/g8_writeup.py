"""Render the completed factorial study from the verified paired tables, with explicit scope for each claim."""
import json
from pathlib import Path

def read(p):return json.loads(p.read_bytes())
def pair(v,d=2):return " / ".join(f"{x:.{d}f}" for x in v)
def stat(v):return f"{v['mean']:+.3f} [{v['paired_ci95'][0]:+.3f}, {v['paired_ci95'][1]:+.3f}]"


def report(folder):
    m=folder/"metrics";d=read(m/"factorial.json");panels=d["panels"];raw=read(m/"raw-and-live.json");cost=read(m/"release.json");arc=read(m/"artifact-inventory.json");ret=read(m/"retention.json")
    gate=d["primary_gate"];npass=sum(v["improves"] for v in d["primary"])
    def rows(task,core,arm,metric):return [panels[task]["series"][f"{core}-{s}"]["cells"][arm][metric] for s in (511,512)]
    def effect(task,core,metric,k):return [panels[task]["series"][f"{core}-{s}"]["paired_contrasts"][metric][k]["mean"] for s in (511,512)]
    hpush=effect("pusht","hybrid","joint_regret_px","other_main");hbar=effect("puck_barrier","hybrid","joint_regret_px","other_main")
    spush=effect("pusht","hybrid","joint_regret_px","sim_main");sbar=effect("puck_barrier","hybrid","joint_regret_px","sim_main")
    word="passed" if gate else "missed"
    lines=["# G8 factorial transfer results","",
      f"The hybrid {word} the prespecified cross-modal decision-transfer criterion. Fresh non-simulator examples changed realized goal regret by {pair(hpush,3)} pixels on new Push-T starts and {pair(hbar,3)} pixels on the previously unseen puck-and-barrier dynamics. Negative changes favor fresh data. The simultaneous one-sided criterion held in {npass} of four required task–seed cells.",
      "",f"Changing simulator starts independently changed hybrid regret by {pair(spush,3)} pixels on Push-T and {pair(sbar,3)} on the puck task. These interventions separate physical training diversity from a bundle of acted speech, audiobooks, text documents and real audiovisual events. Two of the four non-simulator replacements also introduce a domain shift. Every cell still trains on all nine families. These estimates measure the incremental value of replacing examples in the specified families. A separate modality-ablation experiment would be needed to measure the value of including a family at all.",
      "",f"Eight new 116,502,025-parameter continuations reached 16,384 updates. Four 2,048-update bridge trials reproduced the archived inference and optimizer bytes exactly. Eight G7 corners and four G6 parents therefore qualified as controls. Each new fit resumed the same seed-specific parent at 8,192 updates and saw 2,097,152 additional examples at effective batch 256. The single named v5e-1 cost {cost['study_units']:.2f} units. Cumulative G7 and G8 usage was {cost['cumulative_owned_units']:.2f} of the authorized 100. [Protocol](protocol.md) · [Bridge hashes](metrics/bridges.json) · [Training and archive inventory](metrics/artifact-inventory.json)",
      "","## Equal-exposure decisions","",
      "Each cell shows the two training seeds in order. On each zero-shot panel, all models received the same observations, counterfactual candidate plans and independently sampled goals. Regret is excess realized goal distance over the best offered plan. Endpoint error is evaluated separately. Both are in pixels.",
      "","| Panel and metric | Core | G6 parent | Replay | Fresh sim only | Fresh other only | Both fresh |","|---|---|---:|---:|---:|---:|---:|"]
    for t,label in (("pusht","Push-T"),("puck_barrier","Puck + barrier")):
        for what,metric in (("regret","joint_regret_px"),("endpoint","joint_endpoint_px")):
            for core in ("hybrid","gpt2"):
                vals=[pair(rows(t,core,arm,metric)) for arm in ("parent","replay","sim","other","fresh")]
                lines.append(f"| {label} {what} | {'Hybrid' if core=='hybrid' else 'Dense GPT-2'} | "+" | ".join(vals)+" |")
    lines+=["","A separate fixed-pose audit found that removing the thin wall changed zero native camera pixels at all 64 development poses. Area averaging the same render made the wall visible. In another 512 pairs with matching histories and a hidden wall state, an oracle forced to choose the same plan incurred 0.87 pixels of excess regret (95% interval 0.75–1.00). The paired bound concerns that supplementary mixture. The registered puck panel always contains the barrier. [Visibility and paired-world controls](observation-check.md)"]
    lines+=["","## Which data changed the decision?","","The simple effects compare fixed simulator training data. The average non-simulator effect is the mean of other-minus-replay and both-minus-sim. The interaction is both minus sim minus other plus replay. Brackets give 4,000-draw, whole-start paired 95% intervals. The four primary simultaneous upper bounds use the 98.75th percentile and include both hybrid seeds and tasks.",
        "","| Panel | Core and seed | Other effect, old sim | Other effect, new sim | Average other effect | Interaction |","|---|---|---:|---:|---:|---:|"]
    for t,label in (("pusht","Push-T"),("puck_barrier","Puck + barrier")):
        for core in ("hybrid","gpt2"):
            for seed in (511,512):
                e=panels[t]["series"][f"{core}-{seed}"]["paired_contrasts"]["joint_regret_px"]
                lines.append(f"| {label} | {core} {seed} | "+" | ".join(stat(e[k]) for k in ("other_at_replayed_sim","other_at_fresh_sim","other_main","interaction"))+" |")
    lines+=["","![Paired effects of physical and non-simulator training examples](decision-effects.png)",
      "",f"Uniform choice incurred {raw['pusht']['scores']['persistence']['regret_px']:.2f} pixels on Push-T and {raw['puck_barrier']['scores']['persistence']['regret_px']:.2f} on the puck-and-barrier task. A geometry ridge fitted only on original training scenes incurred {raw['pusht']['scores']['geometry']['regret_px']:.2f} and {raw['puck_barrier']['scores']['geometry']['regret_px']:.2f} pixels. The fresh panels contain 2,048 starts and four goals per start each. Full tables include state alone, correct and wrong camera, plan erasure, exact paired hybrid-versus-dense contrasts and the parent for every case. [All contrasts](metrics/factorial.json) · [Per-start primary arrays](metrics/paired-scenes.npz)",
      "","## Transfer to another simulator and action interface","",
      "The MuJoCo Reacher head fits 512 training starts for each frozen field and ranks nine realized torque trajectories on 512 independent starts. Both goals are scored on decisions two through four. This is supervised transfer with a matched label budget. Units are centimetres of regret.",
      "","| Core | Parent | Replay | Fresh sim | Fresh other | Both fresh | Average other effect |","|---|---:|---:|---:|---:|---:|---:|"]
    for core in ("hybrid","gpt2"):
        vals=[pair(rows("reacher",core,arm,"regret_cm")) for arm in ("parent","replay","sim","other","fresh")]
        lines.append(f"| {core} | "+" | ".join(vals+[pair(effect("reacher",core,"regret_cm","other_main"),3)])+" |")
    lines+=["",f"The raw-observation heads incurred {raw['reacher']['raw-511']['regret_cm']:.2f} / {raw['reacher']['raw-512']['regret_cm']:.2f} cm and the matched-data, body-frame physical prior incurred {raw['reacher']['physical']['regret_cm']:.2f} cm. [Raw controls and head measurements](metrics/raw-and-live.json)",
      "","## Sustained operation","","Each controller made 24 consecutive decisions and executed 384 physical ticks. The 64 starts per task and the proposal RNG were paired. The goal-aware proposal generator is shared by every controller. Each prediction uses the latest 16 observed ticks. Report terminal success inside a 20-pixel radius separately from ever reaching that radius.",
      "","| Task | Core | Replay success | Fresh sim | Fresh other | Both fresh | Both fresh terminal distance |","|---|---|---:|---:|---:|---:|---:|"]
    for task,title in (("live_pusht","Push-T"),("live_puck_barrier","Puck + barrier")):
        for core in ("hybrid","gpt2"):
            vals=[pair([100*x for x in rows(task,core,arm,"final_success")],1)+"%" for arm in ("replay","sim","other","fresh")]
            lines.append(f"| {title} | {core} | "+" | ".join(vals+[pair(rows(task,core,"fresh","terminal_distance_px"))+" px"])+" |")
    lines+=["","| Task | First goal-directed proposal | Uniform choice | Passive | Geometry prior | Geometry terminal distance |","|---|---:|---:|---:|---:|---:|"]
    for task,title in (("live_pusht","Push-T"),("live_puck_barrier","Puck + barrier")):
        values=[f"{100*raw[task][name]['success']:.1f}%" for name in ("first_proposal","uniform","passive","geometry")]
        lines.append(f"| {title} | "+" | ".join(values+[f"{raw[task]['geometry']['mean_final_px']:.2f} px"])+" |")
    lines+=["","The previews replay the measured actions on the first predetermined start. Every decision endpoint is compared numerically with its saved trajectory. The red goal overlay was added for viewing. [Push-T replay](examples/pusht.webm) · [Puck-and-barrier replay](examples/puck_barrier.webm) · [Replay hashes and error](metrics/rollout-examples.json) · [Run decision inference on the supplied observation](inference/decisions.md)",
      "","## Other modalities","","This secondary panel reuses the fixed G7 held examples. Byte NLL is lower when next-byte predictions improve. Video-to-audio feature MSE and real-event AP are diagnostic losses; waveform intelligibility was measured independently.",
      "","| Core and seed | Replay byte NLL | Sim only | Other only | Both fresh | Both fresh video→audio MSE | Both fresh Tokyo event AP |","|---|---:|---:|---:|---:|---:|---:|"]
    for core in ("hybrid","gpt2"):
        for seed in (511,512):
            cells=[ret[f"{core}-{seed}-{arm}"] for arm in ("replay","sim","other","fresh")]
            vals=[f"{v['signals']['text_text']['byte_nll']:.4f}" for v in cells];f=cells[-1]
            lines.append(f"| {core} {seed} | "+" | ".join(vals+[f"{f['signals']['crema_video']['audio_mse']:.3f}",f"{f['held_event_ap']['correct']:.5f}"])+" |")
    lines+=["","Separate rate-controlled experiments on fresh readers found a practical acoustic decoder. At 2,250 bits/s, trained bit allocation cut WER from 93.5% to 46.7%. At 9,000 bits/s WER was 22.5%, close to the original audio’s 22.4%. A whole-utterance replication on 40 further readers met the ±3-point recognition margin at 9 kbit/s for two recognizers. The frozen world-state readouts still failed the genuine text/video-to-waveform content criterion. [Speech evidence and examples](speech.md) · [Portable weights and decoder instructions](inference/README.md)",
      "","## What to investigate next",""]
    if gate:lines.append("Replicate the identified non-simulator effect across additional training seeds and then split the four-family bundle. A causal contribution from a bundle does not identify which modality carried it. Require the same decision advantage when the available sensors at inference change.")
    else:lines.append("Keep the physical trajectories matched and add synchronized observations whose information actually changes the best action. Use paired scenes with identical apparent state but a different physical condition, a sensor that reveals that condition, and an oracle showing that choosing differently improves realized return. The present study does not establish replicated cross-modal decision transfer under its registered criterion.")
    lines+=["","Use the demonstrated acoustic codec as a downstream reference while testing shorter-timescale supervision and alternate alignment objectives in the shared field. Require prediction of recoverable words under correct, wrong and erased inputs before optimizing listener preference.","",
      f"All {arc['status']['durable_rungs']} newly trained optimizer and inference generations have independent physical readback. Large files are stored under `/mnt/slwm-checkpoints-a/SLWM/g8` and `/mnt/slwm-checkpoints-b/SLWM/g8`. Confirmation data are on the separate Fedora SSD. [Compute and release receipt](metrics/release.json) · [Complete G7 study](../../t7/continuation-alignment-control-20261110/report.md)"]
    (folder/"report.md").write_text("\n".join(lines)+"\n")
    return {"gate":gate,"passed_cells":npass,"push":hpush,"puck":hbar,"sim_push":spush,"sim_puck":sbar}


def readme(root,summary,cost):
    p=root/"README.md";s=p.read_text();begin=s.index("The G8 action experiment holds physical training trajectories fixed");end=s.index("\n\n",begin)
    outcome="met" if summary["gate"] else "missed"
    text=f"The completed G8 four-cell study {outcome} its prespecified cross-modal decision criterion ({summary['passed_cells']}/4 required hybrid task–seed cases). At fixed simulator freshness, the average non-simulator effect was {pair(summary['push'],3)} pixels of realized goal regret on new Push-T starts and {pair(summary['puck'],3)} on unseen puck-and-barrier dynamics. Lower regret is better. The study includes matched dense Transformers, 512-start articulated-arm readouts and 38.4-second closed-loop action traces. It used {cost['study_units']:.2f} additional project Colab units. [Factorial results](experiments/multimodal/t8/factorial-transfer-20261110/report.md) · [Executed decision replays](experiments/multimodal/t8/factorial-transfer-20261110/metrics/rollout-examples.json)"
    s=s[:begin]+text+s[end:];s=s.replace("The next comparison crosses fresh simulator starts with fresh examples from the other modalities and retains a matched dense Transformer.","G8 independently varied fresh simulator starts and fresh non-simulator examples with a matched dense Transformer.")
    previous="With simulator training trajectories held fixed, do additional speech, text and audiovisual observations improve decisions on a new physical task?"
    new_question="Which non-simulator families drive the replicated decision effect, and does it generalize to additional seeds and sensor access?" if summary["gate"] else "When present state is ambiguous, can synchronized evidence reveal a physical condition that changes the optimal action?"
    s=s.replace(previous,new_question)
    p.write_text(s)

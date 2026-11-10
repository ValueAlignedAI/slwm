"""Integrate an independently complete G8 milestone after exact-tree review and update this study's state."""
import argparse
from pathlib import Path
import json
from g7_integrate_release import integrate,git
from g8_state import update

REPORT=Path("experiments/multimodal/t8/factorial-transfer-20261110")


def main(a):
    if a.kind=="speech":
        doc=json.loads(git("show",a.head+":"+str(REPORT/"metrics/speech.json")))
        if set(doc["allocated"]["cases"])!={"libri","crema"} or any(len(v["variants"])!=4 for v in doc["allocated"]["cases"].values()):
            raise ValueError("Waveform comparison incomplete")
        title="Publish the verified G8 acoustic codec and independent speaker results"
        body="Releases the 651,696-byte trained acoustic model, real packed streams, a 48-second file round trip and matched frozen-model generation controls. At 2,250 bits/s, training-distortion allocation reduced held audiobook WER from 93.5% to 46.7% using the same payload and renderer. At 9,000 bits/s the result was 22.5%, versus 22.4% on source audio. Text- and video-conditioned generation still failed the prespecified content gate. The separate ongoing physical-transfer factorial is documented as active."
    else:
        stamp=json.loads(git("show",a.head+":"+str(REPORT/"metrics/release.json")))
        if not stamp["complete"] or not stamp["all_owned_released"]:raise ValueError("G8 scientific release incomplete")
        title="Publish completed G8 factorial transfer, decision rollouts and research conclusions"
        body="Publishes the registered 116.50M four-cell experiment with matched dense Transformer and raw controls, four byte-exact bridges, independent starts and a new physical task, efficient articulated-arm transfer, and sustained executed decisions. The report includes both seeds, the preregistered gate, whole-start uncertainty, exact weights/optimizer receipts, the project-only ledger, inference outputs and updated README."
    def on_merge(repo,ctl,event,receipt,finished=False):
        update(ctl,event=event,phase="integrated" if finished else "factorial_training_and_validation",receipt=receipt)
    receipt=integrate(a.repo,a.control,a.branch,a.head,title,body,final=a.kind=="factorial",recorder=on_merge)
    print(json.dumps(receipt,sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("kind",choices=("speech","factorial"));p.add_argument("--repo",type=Path,required=True);p.add_argument("--control",type=Path,required=True)
    p.add_argument("--branch",required=True);p.add_argument("--head",required=True);main(p.parse_args())

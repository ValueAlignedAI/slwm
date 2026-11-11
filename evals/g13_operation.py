"""Check portable model parity, genuine waveform substitutions and sustained real chronological clocks."""

import argparse
import json
import time
import jax
import numpy as np
from scripts.infer.g13_world import (
    read,
    model,
    encoded,
    forecast,
    commit,
    select_plan,
    direct_action,
    voice,
)
from scripts.infer.g13_component import infer
from data.g13_goals import DIRECTIONS
from scripts.train.g13_optim import ROOT
from scripts.train.g9_storage import write
from evals.g13_predict import bank

PUB = ROOT / "release"
PKG = PUB / "inference"
EX = PUB / "examples"
MET = PUB / "metrics"


def parity():
    index = json.loads((PKG / "manifest.json").read_bytes())
    r = bank("repaired", "test", True)
    ii = np.arange(83, 83 + 256)
    target = ROOT / "work/score/repaired/fresh"
    trace = {}
    for key, m in index["worlds"].items():
        net = model(PKG, key)
        scene = {x: r[x][ii] for x in ("px", "geo", "kin", "plan", "goal", "bind")}
        o, rows = forecast(scene, net)
        z = read(target / (m["core"] + "-" + m["recipe"] + "-1301.npz"))
        obj = float(np.abs(o["mover"] - z["correct_mover"][ii]).max())
        act = float(np.abs(o["active"][:, -1] - z["correct_active"][ii]).max())
        raw = float(np.abs(o["action"].sum(1) - z["correct_cmd"][ii]).max())
        if max(obj, act, raw) > 1e-5:
            raise ValueError(f"Portable {key} mismatch {obj}/{act}/{raw}")
        altered = dict(scene, plan=-scene["plan"])
        swap, _ = forecast(altered, net)
        pd = float(np.abs(swap["action"] - o["action"]).max())
        md = float(np.abs(swap["mover"] - o["mover"]).max())
        if pd > 1e-6 or md > 1e-6:
            raise ValueError("Future controls entered a passive output")
        trace[key] = dict(
            probability_error=obj,
            active_error_m=act,
            action_error_m=raw,
            plan_to_passive_error=pd,
            plan_to_object_error=md,
        )
    # Verify the task-aligned action readouts against the fourth untouched cohort.
    from evals.g13_direct import data

    final = data("last")
    known = read(MET / "direct-actions.npz")
    jj = np.arange(196, 196 + 256)
    for core in ("gpt2", "transport", "dense_spatial", "associative"):
        net = model(PKG, core + "-linear")
        scene = {x: final[x][jj] for x in ("px", "geo", "kin", "plan", "goal", "bind")}
        p, raw = direct_action(scene, net, PKG)
        err = float(np.max(np.abs(p - known[core + "-linear-1301-protected"][jj])))
        if err > 1e-5:
            raise ValueError(
                f"Direct action packaging changed {core} prediction by {err}"
            )
        trace[core + "-linear"]["direct_action_error_m"] = err
    write(MET / "world-parity.json", trace)
    print("Verified passive independence and all portable robot fields", flush=True)


def audio_and_nouns():
    index = json.loads((PKG / "manifest.json").read_bytes())
    cases = json.loads((EX / "voice-examples.json").read_bytes())
    saved = read(MET / "voice-portable-cases.npz")
    spoken = []
    for row in cases:
        word = row["word"]
        scene0 = read(EX / f"voice-{word}.npz")
        scene = {k: scene0[k] for k in ("px", "geo", "kin", "plan")}
        for core in ("gpt2", "transport"):
            net = model(PKG, core + "-anchored")
            for mode, file in (("correct", "true"), ("wrong", "foil")):
                wav = EX / f"voice-{word}-{file}.wav"
                for target in ("object", "effector"):
                    output = voice(scene, net, PKG, wav, row["goal"], target=target)
                    expected = saved[
                        f"fresh_{core}-anchored-1301_{target}_protected_{mode}"
                    ][row["voice_index"]]
                    err = abs(output["probability"] - float(expected.max()))
                    if (
                        output["forecast"] != DIRECTIONS[int(expected.argmax())]
                        or err > 0.025
                    ):
                        raise ValueError(
                            f"Voice route did not reproduce {core} {word} {mode} {target}: {err}"
                        )
                    spoken.append(
                        dict(
                            word=word,
                            core=core,
                            mode=mode,
                            target=target,
                            probability_error=err,
                            **{k: v for k, v in output.items() if k != "target"},
                        )
                    )
    comps = json.loads((EX / "component-examples.json").read_bytes())
    known = read(MET / "component-predictions.npz")
    out = []
    for row in comps:
        fold = row["fold"]
        data = read(EX / f"component-{fold}.npz")
        for core in ("gpt2", "gated", "oscillator", "associative"):
            for arm in ("semantic", "learned"):
                for mode in ("correct", "wrong", "none"):
                    pred = infer(data, PKG, fold, core, arm, mode, inspect=True)[0]
                    target = known[f"{fold}-{core}-{arm}-1301_{mode}_top"][row["index"]]
                    if pred["proposals"][0]["noun"] != index["nouns"][target[0]]:
                        raise ValueError("Component top proposal mismatch")
                    out.append(dict(fold=fold, core=core, arm=arm, mode=mode, **pred))
    write(MET / "voice-waveforms.json", spoken)
    write(MET / "component-inference.json", out)
    print(
        "Verified real spoken word substitutions and native component packages",
        len(spoken),
        len(out),
        flush=True,
    )


def raw_native():
    rows = json.loads((EX / "component-examples.json").read_bytes())
    out = []
    for row in rows:
        fold = row["fold"]
        scene = read(EX / f"component-{fold}.npz")
        for mode, word in (("correct", "true"), ("wrong", "foil")):
            base = infer(scene, PKG, fold, "gpt2", "semantic", mode, inspect=True)[0]
            result = infer(
                scene,
                PKG,
                fold,
                "gpt2",
                "semantic",
                "correct",
                EX / f"component-{fold}-{word}.wav",
                ROOT.parent / "g11/data/whisper-small",
                ROOT / "data/components/minilm",
                True,
            )[0]
            out.append(
                dict(
                    fold=fold,
                    wav=word,
                    prepared=base,
                    raw=result,
                    same_route=base["route"] == result["route"],
                    same_top=base["proposals"][0]["noun"]
                    == result["proposals"][0]["noun"],
                )
            )
    write(MET / "natural-waveforms.json", out)
    print(
        "Ran native AV sensor history with four fresh waveform interventions",
        flush=True,
    )


def rolling(net, scene, truth, refresh):
    # Beyond the initial prefix, the uncorrected simulation sees only logged command inputs and frozen visual history.
    x = truth["all_measured_positions"]
    u = truth["all_recorded_commands"]
    est = x.copy()
    marg = []
    raw = np.asarray(scene["px"][:1])
    geo = np.asarray(scene["geo"][:1])
    for j, t in enumerate(scene["tick"]):
        if refresh and j and j % refresh == 0:
            est[t - 3 : t + 1] = x[t - 3 : t + 1]
        ix = t - 3 + np.arange(4)
        pts = est[ix]
        previous = np.where((ix > 0)[:, None], u[np.maximum(ix - 1, 0)], 0)
        kin = np.concatenate((pts, np.diff(pts, axis=0, prepend=pts[:1]), previous), 1)
        step = dict(px=raw, geo=geo, kin=kin[None], plan=scene["plan"][j : j + 1])
        predicted = forecast(encoded(step, ""), net)[0]["calibrated"][0, 0]
        est[t + 1] = est[t] + predicted
        marg.append(float(np.linalg.norm(est[t + 1] - x[t + 1]) * 1000))
    return (
        dict(
            mean_mm=float(np.mean(marg)),
            final_mm=marg[-1],
            last_state=est[int(scene["tick"][-1]) + 1].tolist(),
        ),
        est[scene["tick"] + 1].copy(),
    )


def live():
    streams = json.loads((EX / "robot-streams.json").read_bytes())
    results = []
    traces = {}
    for row in streams:
        name = row["stream"]
        scene = read(EX / (name + ".npz"))
        truth = read(EX / (name + "-outcomes.npz"))
        n = len(scene["tick"])
        for which in ("gpt2-linear", "transport-linear"):
            net = model(PKG, which)
            events = []
            nominal = []
            plans = []
            tic = time.monotonic()
            for i, t in enumerate(scene["tick"]):
                raw = {k: scene[k][i : i + 1] for k in ("px", "geo", "kin", "plan")}
                r = encoded(raw, "")
                o, _ = forecast(r, net)
                nominal.append(o["calibrated"][0, -1])
                event = commit(o, net[1])
                events.append(dict(clock=int(t), **event))
                angle = 2 * np.pi * (int(t) % 64) / 64.0
                goal = 0.055 * np.array([np.cos(angle), np.sin(angle)])
                plans.append(select_plan(r, net, goal))
            seconds = time.monotonic() - tic
            err = (
                np.linalg.norm(np.stack(nominal) - truth["future"][:, -1], axis=1)
                * 1000
            )
            free, trajectory = rolling(net, scene, truth, 0)
            resync, recovered = rolling(net, scene, truth, 8)
            key = name + "_" + which
            traces[key + "_free"] = trajectory
            traces[key + "_refresh"] = recovered
            traces[key + "_observed"] = truth["all_measured_positions"][
                scene["tick"] + 1
            ]
            results.append(
                dict(
                    stream=name,
                    model=which,
                    clocks=n,
                    seconds_including_startup=seconds,
                    source_observations=row["observations"],
                    four_step_mm=float(err.mean()),
                    free=free,
                    refresh_eight=resync,
                    internal_only=sum(e["route"] == "internal_only" for e in events),
                    hypothetical_plan_commits=sum(p["route"] == "plan" for p in plans),
                )
            )
            if which == "transport-linear":
                with (EX / (name + "-decisions.jsonl")).open("w") as f:
                    for event, plan in zip(events, plans):
                        f.write(
                            json.dumps(
                                dict(**event, offline_plan=plan), separators=(",", ":")
                            )
                            + "\n"
                        )
    write(MET / "live.json", results)
    np.savez_compressed(MET / "rollout-traces.npz", **traces)
    print("Validated six chronological physical replays", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("parity", "audio", "raw", "live"))
    a = p.parse_args()
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "cache"))
    {"parity": parity, "audio": audio_and_nouns, "raw": raw_native, "live": live}[
        a.stage
    ]()

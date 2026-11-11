"""Run recorded cues during continuous tabletop operation and release synchronized physical replays."""

import json
from pathlib import Path
import subprocess
import wave
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2
from data.g14_table import ROOT, LIMIT, COLORS
from data.g14_grid import SLOTS
from data.g14_wave import pcm
from data.g14_policy import choose
from scripts.train.g14_execute import score
import scripts.data.g14_collect as collector
from scripts.train.g9_storage import write
from evals.g13_stats import Groups


def mission_plan(index):
    cohort = "fresh" if index >= 8 else "held"
    case = json.loads((ROOT / "data/cases" / (cohort + ".json")).read_bytes())[
        index % 8
    ]
    order = np.roll([0, 3, 1, 2, 0, 3], index % 4).tolist()
    return case, [case["commands"][i] for i in order]


def image(t, stations, addresses, c, station, requested, clock, caption):
    im = cv2.resize(t.render(), (640, 360), interpolation=cv2.INTER_NEAREST)
    for j, (point, address) in enumerate(zip(stations, addresses)):
        pos = np.linalg.inv(t.cal[0]) @ np.r_[point, 1.0]
        x, y = np.rint(pos[:2] / pos[2] * [640, 360]).astype(int)
        color = (255, 241, 72) if j == station and c >= 0 else (218, 225, 230)
        code = SLOTS[1][address[0]].upper() + str(address[1])
        cv2.circle(im, (x, y), 6, color, 2)
        cv2.putText(im, code, (x + 7, y - 7), 0, 0.48, color, 1, cv2.LINE_AA)
    buf = np.zeros((442, 640, 3), np.uint8)
    buf[:360] = im
    buf[360:] = (17, 26, 39)

    def put(text, y, scale):
        cv2.putText(buf, text, (14, y), 0, scale, (237, 239, 239), 1, cv2.LINE_AA)

    put(f"{clock:05.1f} s   {caption}", 387, 0.58)
    assigned = (
        COLORS[requested["true_color"]]
        + " to "
        + SLOTS[1][addresses[requested["true_station"]][0]].upper()
        + str(addresses[requested["true_station"]][1])
    )
    put(
        "Assigned: "
        + assigned
        + "   |   Active: "
        + (
            COLORS[c]
            + " to "
            + SLOTS[1][addresses[station][0]].upper()
            + str(addresses[station][1])
            if c >= 0
            else "waiting"
        ),
        419,
        0.55,
    )
    return buf


def movie(folder, name, frames, audio):
    raw = folder / (name + ".wav")
    mp4 = folder / (name + ".mp4")
    with wave.open(str(raw), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    args = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "640x442",
        "-framerate",
        "10/3",
        "-i",
        "pipe:0",
        "-i",
        str(raw),
        "-c:v",
        "libopenh264",
        "-threads",
        "2",
        "-b:v",
        "1050k",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "80k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(mp4),
    ]
    subprocess.run(args, input=np.stack(frames).tobytes(), check=True)


def one(index):
    root = ROOT / "work/missions"
    dest = root / f"mission-{index:02d}.json"
    if dest.is_file():
        return json.loads(dest.read_bytes())
    table = collector.clean_worker()
    seed = 271000 + index
    sig = table.reset(seed, index % 2 == 1)
    stations = table.stations(seed)
    case, seq = mission_plan(index)
    media = json.loads((ROOT / "data/utterances/media.json").read_bytes())
    locks = json.loads((ROOT / "work/final-control-lock.json").read_bytes())
    variant = json.loads((ROOT / "work/geometry-choice.json").read_bytes())["variant"]

    def policy(core):
        return dict(
            core=core,
            arm="protected",
            seed=1411,
            routed=locks["routed"],
            stage=locks["stage"],
            contact_only=locks.get("contact_only", False),
        )

    modes = [
        ("geometry-carry", {"core": "geometry"}),
        ("dense-carry", policy("gpt2")),
        ("spatial-carry", policy("transport")),
        ("spatial-reset", policy("transport")),
        ("spatial-stale", policy("transport")),
        ("spatial-clear-rejected", policy("transport")),
    ]
    waves = {int(c["clip"]): pcm(media[c["clip"]]) for c in seq}
    case_rows = {}
    for name, pol in modes:
        table = collector.clean_worker()
        if table.reset(seed, index % 2 == 1) != sig:
            raise ValueError("Continuous policy did not start from the canonical state")
        if table.physical_signature() != sig:
            raise ValueError("Mission root mismatch")
        c, st = -1, 0
        rows = []
        clock = 0
        frames = []
        audio = []
        tr = []
        eff = []
        chosen = []
        sources = []
        keep = name == "spatial-carry" and index in (0, 8)
        for n, command in enumerate(seq):
            wave0 = waves[command["clip"]]
            listen = max(1, int(np.ceil(len(wave0) / 4800)))
            duration = listen + LIMIT
            buf = np.zeros(duration * 4800, "f4")
            buf[: len(wave0)] = wave0
            if keep:
                audio.append(buf)
            phase = []
            phase_initial = None
            start_clock = clock
            old = (c, st)
            update = False
            for tick in range(duration):
                if tick == listen:
                    phase_initial = table.positions()
                    if command["accept"] and (name != "spatial-stale" or c < 0):
                        c, st = command["heard_color"], command["heard_station"]
                        update = True
                    elif not command["accept"] and name == "spatial-clear-rejected":
                        c = -1
                if name == "spatial-reset" and tick != listen:
                    c = -1
                obs = table.observation()
                if keep:
                    frames.append(
                        image(
                            table,
                            stations,
                            case["addresses"],
                            c,
                            st,
                            command,
                            clock * 0.3,
                            "Listening" if tick < listen else "Goal retained",
                        )
                    )
                cmd, info = choose(table, obs, c, stations[st], pol, variant)
                q, ee = table.execute(cmd)
                if tick >= listen:
                    phase.append(q)
                if keep:
                    tr.extend(q)
                    eff.extend(ee)
                    chosen.append(info["chosen"])
                    sources.append([n, tick, c, st])
                clock += 1
            value = score(
                np.concatenate(phase),
                phase_initial,
                command["true_color"],
                stations[command["true_station"]],
            )
            rows.append(
                dict(
                    phase=n,
                    clip=command["key"],
                    assigned=[command["true_color"], command["true_station"]],
                    previous=list(old),
                    update=update,
                    accepted=command["accept"],
                    recognition=[command["heard_color"], command["heard_station"]],
                    listen_macros=listen,
                    physical_clock=start_clock,
                    **value,
                )
            )
        if keep:
            title = f"mission-{index:02d}"
            movie(root, title, frames, np.concatenate(audio))
            np.savez_compressed(
                root / (title + ".npz"),
                objects=np.asarray(tr),
                effector=np.asarray(eff),
                chosen=chosen,
                active=sources,
                station=stations,
            )
        case_rows[name] = dict(duration_seconds=clock * 0.3, phases=rows)
    # Already-satisfied and missing-goal probes retain native measurements through 24 macros.
    probes = {}
    for mode in ("spatial", "dense", "geometry", "absent"):
        table = collector.clean_worker()
        if table.reset(seed, index % 2 == 1) != sig:
            raise ValueError("Continuous policy did not start from the canonical state")
        initial = table.positions()
        color = seq[0]["true_color"]
        goal = initial[color]
        actions = []
        trajectory = []
        pol = (
            policy("transport" if mode in ("spatial", "absent") else "gpt2")
            if mode != "geometry"
            else {"core": "geometry"}
        )
        for tick in range(LIMIT):
            obs = table.observation()
            cmd, info = choose(
                table, obs, -1 if mode == "absent" else color, goal, pol, variant
            )
            actions.append(np.linalg.norm(cmd))
            trajectory.extend(table.execute(cmd)[0])
        probes[mode] = dict(
            unchanged=float(np.mean(np.asarray(actions) < 1e-7)),
            effort_m=float(np.sum(actions)),
            **score(np.stack(trajectory), initial, color, goal),
        )
    out = dict(
        index=index,
        world=seed,
        source_case=case["seed"],
        speaker=case["speaker"],
        start=sig,
        cohort="fresh" if index >= 8 else "held",
        sequences=case_rows,
        already=probes,
    )
    write(dest, out)
    return out


def main():
    root = ROOT / "work/missions"
    root.mkdir(exist_ok=True)
    with ProcessPoolExecutor(3, initializer=collector.init_worker) as pool:
        rows = list(pool.map(one, range(16)))
    summary = {}
    ids = [r["world"] for r in rows]
    bs = Groups(ids)
    for name in rows[0]["sequences"]:
        get = lambda k: np.array(
            [np.mean([x[k] for x in r["sequences"][name]["phases"]]) for r in rows]
        )
        summary[name] = {
            k: bs.score(get(k))
            for k in ("success", "arrival", "final_mm", "other_mm", "update")
        }

    def efficacy(r, mode):
        return np.mean([x["success"] for x in r["sequences"][mode]["phases"]])

    paired = {
        mode: bs.score([efficacy(r, "spatial-carry") - efficacy(r, mode) for r in rows])
        for mode in (
            "dense-carry",
            "spatial-reset",
            "spatial-stale",
            "spatial-clear-rejected",
            "geometry-carry",
        )
    }
    totals = {
        k: float(np.mean([r["already"][k]["unchanged"] for r in rows]))
        for k in rows[0]["already"]
    }
    write(
        root / "summary.json",
        dict(
            missions=16,
            phases_per_mission=6,
            spatial_over=paired,
            performance=summary,
            already_at_goal_noop_fraction=totals,
            durations=[
                r["sequences"]["spatial-carry"]["duration_seconds"] for r in rows
            ],
            condition="All modes hear each full recording before the update. The active prior goal is retained during speech except in the explicit reset arm.",
        ),
    )
    print("Sustained missions", summary["spatial-carry"]["success"], paired, flush=True)


if __name__ == "__main__":
    main()

"""Recover contiguous real-robot clocks only from verified overlapping camera/state frames."""

from collections import defaultdict
import json
import hashlib
import numpy as np
from g13_acquire import ROOT, write
from g13_robot import visual
from g13_repair import get_frames, decode, components


def main():
    root = ROOT / "robot-prospective"
    out = ROOT.parent / "release/examples"
    out.mkdir(exist_ok=True, parents=True)
    episodes = json.loads((root / "banks/episodes.json").read_bytes())
    by_shard = defaultdict(list)
    for i, r in enumerate(episodes):
        by_shard[r["shard"]].append((i, r))
    graphs = defaultdict(dict)
    incoming = defaultdict(lambda: defaultdict(set))
    for shard, items in by_shard.items():
        with np.load(root / "decoded" / (shard + ".npz")) as z:
            keys = z["frame"]
            pos = z["x"]
        for i, r in items:
            group = r["group"]
            st = r["start"]
            n = r["length"]
            seq = [
                (bytes(keys[j]).decode(), pos[j].tobytes().hex())
                for j in range(st, st + n)
            ]
            for t in range(1, len(seq) - 1):
                a = (seq[t - 1], seq[t])
                b = (seq[t], seq[t + 1])
                if a[0] == a[1] or b[0] == b[1]:
                    continue
                # Multiple mentions of an identical directed edge are the same observed transition.
                graphs[group].setdefault(a, {}).setdefault(b, (i, t))
                incoming[group][b].add(a)
    longest = {}
    for g, edges in graphs.items():
        best = []
        for a in edges:
            parents = incoming[g][a]
            if len(parents) == 1 and len(edges.get(next(iter(parents)), {})) == 1:
                continue
            cur = a
            used = set()
            trace = []
            while len(edges.get(cur, {})) == 1 and cur not in used:
                used.add(cur)
                nxt, ref = next(iter(edges[cur].items()))
                trace.append((cur, nxt, ref))
                cur = nxt
                if len(incoming[g][cur]) != 1:
                    break
            if len(trace) > len(best):
                best = trace
        if best:
            longest[g] = best
    selected = sorted(longest, key=lambda g: (-len(longest[g]), g))[:3]
    index = []
    for k, g in enumerate(selected):
        path = longest[g]
        src = {}
        for _, _, (i, t) in path:
            if i not in src:
                rec = episodes[i]
                frames = get_frames(root, rec)
                with np.load(root / "decoded" / (rec["shard"] + ".npz")) as z:
                    st = rec["start"]
                    n = rec["length"]
                    x = z["x"][st : st + n]
                    u = z["u"][st : st + n]
                src[i] = (frames, x, u)
        px = []
        geo = []
        xs = []
        us = []
        proof = []

        def append(i, t, node):
            images, x, u = src[i]
            frame = images[t]
            if (
                hashlib.blake2b(frame, digest_size=12).hexdigest() != node[0]
                or x[t].tobytes().hex() != node[1]
            ):
                raise ValueError("Broken real source chain")
            raw = decode(frame)
            pix, _ = visual(raw)
            geometry, _ = components(raw)
            px.append(pix)
            geo.append(geometry)
            xs.append(x[t])
            us.append(u[t])
            proof.append(dict(episode=i, source_tick=t, frame=node[0]))

        for node, _, (i, t) in path:
            append(i, t, node[1])
        _, last, (i, t) = path[-1]
        append(i, t + 1, last[1])
        x = np.stack(xs)
        u = np.stack(us)
        p0 = np.stack(px)
        gm = np.stack(geo)
        ticks = np.arange(3, len(x) - 4)
        obs = {key: [] for key in ("px", "geo", "kin", "plan")}
        future = []
        for t in ticks:
            ix = t - 3 + np.arange(4)
            prev = np.maximum(0, ix - 1)
            h = x[ix]
            # Identical causal window convention to training, including zero first in-window displacement.
            kin = np.concatenate(
                (
                    h,
                    np.diff(h, axis=0, prepend=h[:1]),
                    np.where((ix > 0)[:, None], u[prev], 0),
                ),
                1,
            )
            for key, val in (
                ("px", p0[ix]),
                ("geo", gm[ix]),
                ("kin", kin),
                ("plan", u[t : t + 4]),
            ):
                obs[key].append(val)
            future.append(x[t + 1 : t + 5] - x[t])
        tag = f"robot-stream-{k+1}"
        np.savez_compressed(
            out / (tag + ".npz"),
            **{key: np.asarray(v) for key, v in obs.items()},
            tick=ticks,
        )
        np.savez_compressed(
            out / (tag + "-outcomes.npz"),
            future=np.asarray(future),
            all_measured_positions=x,
            all_recorded_commands=u,
        )
        write(out / (tag + "-lineage.json"), dict(group=g, frames=proof))
        index.append(
            dict(
                stream=tag,
                parent=g,
                observations=len(x),
                forecast_clocks=len(ticks),
                source_subtrajectories=len(src),
            )
        )
    write(out / "robot-streams.json", index)
    print("Continuous recorded clocks", index, flush=True)


if __name__ == "__main__":
    main()

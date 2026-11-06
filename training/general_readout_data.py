"""Typed G1 observations and held-out targets for registered frozen readouts.

Targets are returned beside observations. Actor observations never contain the
current command or any future plan, including in alternative feature baselines.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from training.general_batch import base_observation, normalize, rotation6d
from training.general_data import CACHE, ROOT, pose_delta

COMMON = np.array([*range(6), *range(9, 15), *range(18, 24)])
ROBOT_CHANNELS = np.arange(33, 50)


def majority_activity(labels):
    active = labels[labels > 0]
    return int(np.bincount(active, minlength=13).argmax()) if len(active) else 0


def observed_robot_history(trajectory, start, context, stats):
    indices = start + np.arange(context)
    state = rotation6d(trajectory["observed"][indices], trajectory["gripper"][indices])
    previous = np.maximum(indices - 1, 0)
    commands = np.concatenate([
        pose_delta(trajectory["observed"][indices], trajectory["commanded"][previous]),
        trajectory["action"][previous, 6:]], axis=1)
    numeric = np.zeros((context, 64), np.float32)
    present = np.zeros_like(numeric, bool)
    numeric[:, 33:43] = normalize(state, stats["numeric_mean"][33:43], stats["numeric_scale"][33:43])
    numeric[:, 43:50] = normalize(commands, stats["action_mean"], stats["action_scale"])
    present[:, 33:50] = True
    present[indices == 0, 43:50] = False
    numeric[~present] = 0
    return numeric, present, commands


class ReadoutData:
    def __init__(self, cfg, family, role, allow_final=False):
        if role == "test" and not allow_final:
            raise RuntimeError("Final observations are locked until all compared weights and rules freeze")
        self.cfg, self.family, self.role = cfg, family, role
        manifest = json.loads((CACHE / "evaluation_cases.json").read_text())
        registered = [r for r in manifest["cases"][family] if r["role"] == role]
        self.cases = [{**row, "start":start} for row in registered for start in row["starts"]]
        if family == "droid":
            path = ROOT / cfg["robot_family"] / "raw_metadata/scene_role_plan.json"
            rows = json.loads(path.read_text())["rows"]
        else:
            rows = json.loads((CACHE / family / "rows.json").read_text())
        self.rows = {r["id"]:r for r in rows if r["role"] == role}
        self.stats = np.load(CACHE / "normalizers.npz", allow_pickle=False)

    @lru_cache(maxsize=32)
    def load(self, path):
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def __len__(self):
        return len(self.cases)

    def batch(self, indices, goal_control="provided"):
        cases = [self.cases[int(i)] for i in indices]
        if self.family == "mhealth":
            return self._sensor(cases)
        return self._robot(cases, goal_control)

    def _sensor(self, cases):
        t, q = self.cfg["context"], self.cfg["horizon"]
        values, masks, clocks, targets, target_masks, activities = [], [], [], [], [], []
        last, previous = [], []
        for case in cases:
            row, start = self.rows[case["id"]], case["start"]
            data = self.load(str(CACHE / "mhealth" / row["file"]))
            raw = np.asarray(data[start:start+t+q], np.float32).copy()
            # This is the primary cross-source test: unmatched magnetic,
            # temperature, heart-rate and ECG channels are genuinely missing.
            keep = np.zeros(33, bool); keep[COMMON] = True
            raw[:, ~keep] = np.nan
            numeric = np.full((t+q, 64), np.nan, np.float32)
            numeric[:, :33] = normalize(raw, self.stats["numeric_mean"][:33], self.stats["numeric_scale"][:33])
            values.append(np.nan_to_num(numeric[:t])); masks.append(np.isfinite(numeric[:t]))
            targets.append(raw[t:, COMMON]); target_masks.append(np.isfinite(raw[t:, COMMON]))
            last.append(raw[t-1,COMMON]);previous.append(raw[t-2,COMMON])
            clock = np.asarray(self.load(str(CACHE / "mhealth" / row["time_file"]))[start:start+t+q])
            clocks.append(clock-clock[0])
            activity = self.load(str(CACHE / "mhealth" / row["activity_file"]))[start:start+t]
            activities.append(majority_activity(activity))
        obs = base_observation(len(cases), t, .02)
        obs.update(time=np.asarray(clocks, np.float32)[:, :t], numeric=np.stack(values), numeric_present=np.stack(masks))
        return {"obs":obs, "activity":np.array(activities, np.int64),
                "future":np.nan_to_num(np.stack(targets)), "future_present":np.stack(target_masks),
                "last_physical":np.nan_to_num(np.stack(last)), "last_present":np.isfinite(np.stack(last)),
                "previous_physical":np.nan_to_num(np.stack(previous)),
                "groups":[c["person"] for c in cases], "ids":[c["id"]+"@"+str(c["start"]) for c in cases],
                "common":np.zeros((len(cases), 0), np.float32)}

    def _robot(self, cases, goal_control):
        t, q = self.cfg["context"], self.cfg["horizon"]
        values, masks, images, targets, commands, plans, future, anchors, previous = [], [], [], [], [], [], [], [], []
        captions, goals = [], []
        for case in cases:
            row, start = self.rows[case["id"]], case["start"]
            folder = CACHE / self.cfg["robot_family"] / row["id"]
            trajectory = self.load(str(folder / "trajectory.npz"))
            numeric, present, past = observed_robot_history(trajectory, start, t, self.stats)
            values.append(numeric); masks.append(present)
            image = self.load(str(folder / "rgb.npy"))[start:start+t]
            images.append(np.asarray(image, np.float32).transpose(0,1,4,2,3)/127.5-1)
            current = start+t-1
            targets.append(trajectory["action"][current]); previous.append(past[-1])
            anchor = trajectory["observed"][current]; anchors.append(anchor)
            command = trajectory["commanded"][current:current+q]
            grip = trajectory["action"][current:current+q, 6:]
            plans.append(np.concatenate([pose_delta(np.repeat(anchor[None], q, axis=0), command), grip], 1))
            commands.append(command)
            observed = trajectory["observed"][current+1:current+q+1]
            observed_grip = trajectory["gripper"][current+1:current+q+1]
            future.append(np.concatenate([observed, observed_grip], 1))
            captions.append(row["captions"][0] if row["captions"] else "")
            goals.append(bool(row["captions"]))
        obs = base_observation(len(cases), t, 1., True)
        obs.update(numeric=np.stack(values), numeric_present=np.stack(masks), images=np.stack(images), image_present=np.ones((len(cases),t,3),bool))
        if goal_control == "missing":captions = [""]*len(captions)
        elif goal_control == "shuffled":captions = captions[1:]+captions[:1]
        elif goal_control != "provided":raise ValueError(goal_control)
        return {"obs":obs, "captions":captions, "goal_present":np.array(goals,bool),
                "target":np.stack(targets).astype(np.float32), "previous_command":np.stack(previous).astype(np.float32),
                "plans":np.stack(plans).astype(np.float32), "commanded_poses":np.stack(commands).astype(np.float32),
                "future_poses":np.stack(future).astype(np.float32), "anchors":np.stack(anchors).astype(np.float32),
                "common":obs["numeric"][:, -1, ROBOT_CHANNELS],
                "groups":[c["group"] for c in cases], "ids":[c["id"]+"@"+str(c["start"]) for c in cases]}

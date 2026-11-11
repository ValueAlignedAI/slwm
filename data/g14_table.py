"""Official Language-Table dynamics, calibrated camera sensing and physically executed candidate controls."""

import copy
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path("/mnt/slwm-data/SLWM/g14")
COLORS = ("red", "blue", "green", "yellow")
CENTER = np.array([0.375, 0.0])
CHUNK = 3
LIMIT = 24
TOL = 0.025
OTHER = 0.035
K = 8


def clip(x, r):
    return x * min(1.0, r / max(np.linalg.norm(x), 1e-8))


def unit(x):
    return x / max(np.linalg.norm(x), 1e-8)


def typed(image):
    import cv2

    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    H, W = h.shape
    masks = (
        (h < 10) | (h > 167),
        (h > 99) & (h < 137),
        (h > 40) & (h < 91),
        (h > 16) & (h < 39),
    )
    out = np.zeros((4, 5), np.float32)
    for c, m in enumerate(masks):
        mask = ((s > 92) & (v > 45) & m).astype("u1")
        n, lab, st, ce = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cand = []
        for i, (x, y, w, hh, a) in enumerate(st[1:], 1):
            if (
                5 <= a <= 0.025 * H * W
                and w >= 2
                and hh >= 2
                and x > 0
                and y > 0
                and x + w < W
                and y + hh < H
            ):
                cand.append((a, ce[i, 0], ce[i, 1], w, hh))
        if cand:
            a, x, y, w, hh = max(cand)
            out[c] = (x / W, y / H, w / W, hh / H, 1)
    return out


def pose_digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.asarray(a, dtype="<f8").tobytes())
    return h.hexdigest()


class Table:
    def __init__(self, hz=10):
        from language_table.environments import blocks, language_table

        self.e = language_table.LanguageTable(
            block_mode=blocks.LanguageTableBlockVariants.BLOCK_4,
            render_text_in_image=False,
            control_frequency=float(hz),
            seed=14000,
        )
        self.p = self.e.pybullet_client
        self.names = tuple(
            next(b for b in self.e._blocks_on_table if b.startswith(c + "_"))
            for c in COLORS
        )
        self.ids = [self.e._block_to_pybullet_id[b] for b in self.names]
        self.v, self.proj, *_ = self.e.calc_camera_params((180, 320))
        self.nominal = [self.p.getDynamicsInfo(i, -1)[:2] for i in self.ids]
        self.actuator = self.e._robot.xarm
        self.joints = list(self.e._robot._joints)
        self.map = np.arange(4)
        self.last = np.tile(CENTER, (4, 1))
        self.previous_image = None
        self.cal = None
        self.history = []
        self.last_u = np.zeros(2)
        self.root_state = None

    def positions(self):
        return np.asarray(
            [
                self.p.getBasePositionAndOrientation(self.ids[i])[0][:2]
                for i in self.map
            ],
            np.float32,
        )

    def effector(self):
        return np.asarray(
            self.e._robot.forward_kinematics().translation[:2], np.float32
        )

    def motor(self):
        return np.asarray(self.e._target_effector_pose.translation[:2], np.float32)

    def render(self):
        x = self.p.getCameraImage(
            160,
            90,
            viewMatrix=self.v,
            projectionMatrix=self.proj,
            renderer=self.p.ER_TINY_RENDERER,
            flags=self.p.ER_NO_SEGMENTATION_MASK,
        )
        return np.asarray(x[2], dtype=np.uint8).reshape(90, 160, 4)[:, :, :3]

    def physical_signature(self):
        parts = [self.motor(), self.effector()]
        for i in self.ids:
            parts.extend(self.p.getBasePositionAndOrientation(i))
            parts.extend(self.p.getBaseVelocity(i))
            parts.append(self.p.getDynamicsInfo(i, -1)[:2])
        js = self.p.getJointStates(self.actuator, self.joints)
        parts.append([(v[0], v[1]) for v in js])
        return pose_digest(parts)

    def reset(self, seed, shift=False):
        if self.root_state is not None:
            self.p.removeState(self.root_state[0])
            self.root_state = None
        self.e._rng = np.random.RandomState(seed)
        for i, (m, mu) in zip(self.ids, self.nominal):
            self.p.changeDynamics(i, -1, mass=m, lateralFriction=mu)
        self.e.reset()
        self.map = np.arange(4)
        rng = np.random.default_rng(seed + 140310)
        if shift:
            self.map = rng.permutation(4)
            for i in self.ids:
                m, mu = self.nominal[self.ids.index(i)]
                self.p.changeDynamics(
                    i,
                    -1,
                    mass=m * rng.uniform(0.70, 1.55),
                    lateralFriction=mu * rng.uniform(0.55, 1.65),
                )
        color = (
            (0.86, 0.10, 0.11, 1),
            (0.11, 0.29, 0.92, 1),
            (0.12, 0.79, 0.22, 1),
            (0.94, 0.79, 0.09, 1),
        )
        for c, i in enumerate(self.map):
            self.p.changeVisualShape(self.ids[i], -1, rgbaColor=color[c])
        self.last = np.tile(CENTER, (4, 1))
        self.history = []
        self.last_u = np.zeros(2, np.float32)
        self.root_state = self.save()
        return self.physical_signature()

    def save(self):
        return (
            self.p.saveState(),
            copy.deepcopy(self.e._target_effector_pose),
            self.last.copy(),
            list(self.history),
            self.last_u.copy(),
        )

    def restore(self, state):
        handle, pose, last, hist, u = state
        self.p.restoreState(handle)
        self.e._target_effector_pose = copy.deepcopy(pose)
        # Bullet restores the joints but its motor controller can retain the last command. Reapply the saved setpoint.
        self.e._robot.set_target_effector_pose(self.e._target_effector_pose)
        self.last = last.copy()
        self.history = list(hist)
        self.last_u = u.copy()

    def stations(self, seed):
        q = self.positions()
        rng = np.random.default_rng(seed + 9131)
        points = []
        for center in np.array(
            [[0.29, -0.14], [0.29, 0.14], [0.47, -0.14], [0.47, 0.14]]
        ):
            for trial in range(4000):
                candidate = np.clip(
                    center + rng.uniform(-0.065, 0.065, 2),
                    [0.235, -0.22],
                    [0.515, 0.22],
                )
                if np.linalg.norm(q - candidate, axis=1).min() < 0.058:
                    continue
                if (
                    points
                    and np.linalg.norm(np.asarray(points) - candidate, axis=1).min()
                    < 0.12
                ):
                    continue
                points.append(candidate)
                break
            else:
                raise RuntimeError(
                    "Unable to place a separated free station in this world"
                )
        return np.asarray(points)[rng.permutation(4)].astype("f4")

    def observation(self):
        import cv2

        image = self.render()
        d = typed(image)
        points = self.last.copy()
        if self.cal is None:
            raise ValueError("Camera needs training calibration")
        for c in range(4):
            if d[c, 4]:
                v = self.cal[c] @ np.r_[d[c, :2], 1.0]
                points[c] = v[:2] / v[2]
        self.last = points
        px = cv2.resize(image, (16, 10), interpolation=cv2.INTER_AREA).ravel()
        obs = dict(
            q=points.astype("f4"),
            seen=d[:, 4].astype("f4"),
            uv=d.astype("f4"),
            p=self.effector(),
            motor=self.motor(),
            px=px,
            u=self.last_u.copy(),
        )
        self.history = (self.history + [obs])[-4:]
        while len(self.history) < 4:
            self.history.insert(0, obs)
        return obs

    def execute(self, cmd):
        cmd = np.asarray(cmd, dtype="f4")
        if (
            cmd.shape != (2,)
            or not np.isfinite(cmd).all()
            or np.linalg.norm(cmd) > 0.106
        ):
            raise ValueError("Invalid macro control")
        u = cmd / CHUNK
        poses = []
        eef = []
        for _ in range(CHUNK):
            self.e._step_robot_and_sim(u)
            poses.append(self.positions())
            eef.append(self.effector())
        self.last_u = cmd.copy()
        return np.asarray(poses), np.asarray(eef)


def geometric(obs, color, target, route=True, variant=0):
    q = np.asarray(obs["q"][color])
    p = np.asarray(obs["p"])
    motor = np.asarray(obs["motor"])
    d = target - q
    dist = np.linalg.norm(d)
    if dist < 0.014:
        return np.zeros(2, "f4")
    axis = unit(d)
    side = np.array([-axis[1], axis[0]])
    r = p - q
    along = float(r @ axis)
    across = float(r @ side)
    # Stage around an object before pushing. A small touch corridor prevents advancing on the wrong side.
    behind = 0.055 if variant == 0 else 0.047
    if not route:
        desired = q + axis * min(0.068, dist)
    elif -0.092 < along < 0.014 and abs(across) < (0.014 if variant == 0 else 0.019):
        desired = q + axis * min(0.055 if variant == 0 else 0.046, dist)
    else:
        desired = q - behind * axis
        seg = desired - p
        z = np.clip((q - p) @ seg / max(1e-8, seg @ seg), 0, 1)
        if np.linalg.norm(p + z * seg - q) < 0.038 and along > -0.025:
            sign = 1 if across >= 0 else -1
            # A tangent waypoint keeps the approach on the clear side of the target object.
            middle = q + sign * side * 0.060 - axis * (0.006 if along > 0 else 0.032)
            desired = middle
    return clip(desired - motor, 0.052 if variant == 0 else 0.040).astype("f4")


def options(obs, color, target, variant=0):
    base = geometric(obs, color, target, True, variant)
    u = unit(base)
    side = np.array([-u[1], u[0]])
    return np.stack(
        [
            np.zeros(2),
            base,
            0.55 * base,
            clip(1.45 * base, 0.075),
            clip(base + 0.015 * side, 0.060),
            clip(base - 0.015 * side, 0.060),
            geometric(obs, color, target, True, 1 - variant),
            geometric(obs, color, target, False, variant),
        ]
    ).astype("f4")


def loss_cost(q, p, color, target):
    axis = unit(target - q[color])
    behind = q[color] - 0.040 * axis
    return float(np.linalg.norm(q[color] - target) + 0.23 * np.linalg.norm(p - behind))


def utility(q0, p0, q1, p1, color, target, cmd):
    side = np.delete(np.linalg.norm(q1 - q0, axis=1), color)
    return (
        loss_cost(q0, p0, color, target)
        - loss_cost(q1, p1, color, target)
        - 0.7 * side.max()
        - 0.012 * np.linalg.norm(cmd)
    )


def evaluate_path(poses, initial, color, target):
    target_dist = np.linalg.norm(poses[:, color] - target, axis=1)
    off = np.linalg.norm(poses - initial[None], axis=2)
    off = np.delete(off, color, axis=1)
    return target_dist, off.max(1)

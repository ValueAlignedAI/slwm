# SLWM-X1B (`1.008B`) Closed-Loop Mission Replays & Filmstrips

Generated deterministically from SHA-256-verified `1,008,454,410`-parameter checkpoints via:

```bash
.venv/bin/python scripts/eval/x1b_infer.py --build-study-pack
```

## Verified Contents (`s2001` `1.008B` Cohort)

- [`closed_loop_replays.json`](closed_loop_replays.json) — Tick-by-tick `64x64` physical puck (`obj_xy`), pusher (`ee_xy`), commanded motor velocity (`action_xy`), decoded working-memory goal (`latched_goal_xy`), visual `1-px` barrier probability, `PolicyCommitGate` mode routing under `180°` camera spoofing, and live `T=1024` (`24,576`-slot) associative overwrite recall checks across `SLWM-X1B`, `DenseTransformer-X1B`, and `LegacyScalar-X1B`.
- [`preview_thin_barrier_mpc.png`](preview_thin_barrier_mpc.png) — Side-by-side `64x64` rendered frames (`t = 0, 8, 18, 28, 37`) and continuous 2D detour trajectories around the `1-px` (`x=0`) central barrier (`SLWM-X1B` `PASS` at `d = 0.01`).
- [`preview_ood_goal_switch_mpc.png`](preview_ood_goal_switch_mpc.png) — Side-by-side `64x64` rendered frames (`t = 2, 24, 37, 54, 75`) on the OOD `T=76` (`2.38× T_train`) mid-episode Goal-B overwrite mission with zero external goal input (`SLWM-X1B` `PASS` at `d = 0.03` vs `DenseTransformer-X1B` `FAIL` at `d = 0.58` and `LegacyScalar-X1B` `FAIL` at `d = 0.60`).

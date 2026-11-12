# SLWM-X1B (`1.008B`) Closed-Loop Mission Replays & Filmstrips

Generated deterministically from SHA-256-verified `1,008,454,410`-parameter checkpoints via:

```bash
.venv/bin/python scripts/eval/x1b_infer.py --build-study-pack
```

## Contents (Populated Upon Wave 4 `1.008B` Cohort Verification)

- `closed_loop_replays.json` — Tick-by-tick `64x64` physical puck (`obj_xy`), pusher (`ee_xy`), commanded motor velocity (`action_xy`), decoded working-memory goal (`latched_goal_xy`), visual `1-px` barrier probability, `PolicyCommitGate` mode routing under `180°` camera spoofing, and live `T=1024` (`24,576`-slot) associative overwrite recall checks across `SLWM-X1B`, `DenseTransformer-X1B`, and `LegacyScalar-X1B`.
- `preview_thin_barrier_mpc.png` — Side-by-side `64x64` rendered frames (`t = 0, 8, 18, 28, 37`) and continuous 2D detour trajectories around the `1-px` (`x=0`) central barrier.
- `preview_ood_goal_switch_mpc.png` — Side-by-side `64x64` rendered frames (`t = 2, 24, 37, 54, 75`) on the OOD `T=76` (`2.38× T_train`) mid-episode Goal-B overwrite mission with zero external goal input.

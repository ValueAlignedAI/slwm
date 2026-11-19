"""Unit tests for SLWM-v3 Gate 1 observability, decoupled oracle matrix, shortcut-free counterfactuals, and power."""

from __future__ import annotations

import numpy as np
import pytest

from evals.v3_gate1_oracle_and_observability import (
    compute_crossed_design_power,
    compute_paired_power_requirements,
    generate_paired_delayed_correction_streams,
    run_gate1_2x2_oracle_control_matrix,
    stacked_sliding_window_receptive_field,
)


def test_paired_and_crossed_power_calculations():
    res = compute_paired_power_requirements(effect_pp=5.0, discordant_rate=0.25, alpha=0.05, power=0.80)
    assert 775 <= res["n_independent_pairs"] <= 795
    assert res["design_effect_deff"] == pytest.approx(1.0)

    # 8 speakers at N=1024, ICC=0.10 must expose severe clustering (DEFF=13.7, N_eff ~ 75)
    g8 = compute_crossed_design_power(n_episodes=1024, n_clusters_a=8, icc_a=0.10, discordant_rate=0.25)
    assert g8["deff"] == pytest.approx(13.7, abs=0.05)
    assert g8["n_eff"] < 80.0

    # 250 unseen speakers (SpeechCommands) at N=1024, ICC=0.10 yields N_eff > 780 and MDE <= 5.1 pp
    sc = compute_crossed_design_power(n_episodes=1024, n_clusters_a=250, icc_a=0.10, discordant_rate=0.25)
    assert sc["n_eff"] > 780.0
    assert sc["mde_80pct_pp"] <= 5.1

    # 1024 independent unseen control worlds at p_disc=0.10 achieves null CI half-width < 2.0 pp!
    ctrl_p = compute_crossed_design_power(n_episodes=1024, n_clusters_a=1024, icc_a=0.0, discordant_rate=0.10)
    assert ctrl_p["null_ci95_half_width_pp"] < 2.0


def test_stacked_receptive_field_formula():
    assert stacked_sliding_window_receptive_field(24, 64) == 1512
    assert stacked_sliding_window_receptive_field(16, 64) == 1008
    assert stacked_sliding_window_receptive_field(12, 32) == 372


def test_counterfactual_streams_randomized_shortcut_free_and_scale_parameterized():
    for layers, win in ((12, 32), (16, 64)):
        out = generate_paired_delayed_correction_streams(
            seed=42, batch_size=32, comparator_layers=layers, comparator_window=win
        )
        audit = out["audit"]
        assert audit["gate1_counterfactual_pass"] is True
        assert audit["strictly_beyond_receptive_field"] is True
        assert audit["overwrite_after_local_window_eviction"] is True
        assert audit["all_post_cue_tensors_bit_exact_equal"] is True
        assert audit["counterfactual_cue_differs_all_modalities"] is True
        assert audit["t_corr_unique_count"] > 5
        assert audit["t_impulse_unique_count"] > 5
        assert 0.15 < audit["init_order_frac_f1_first"] < 0.85
        assert 0.15 < audit["update_target_balance_frac_fact1"] < 0.85
        assert audit["bg_text_nonzero_fraction"] == 1.0
        for k, v in audit["post_cue_max_abs_diff_by_tensor"].items():
            assert v == 0.0, f"Post-cue leak in {k}: {v}"
        # Clean vs stress streams must differ strictly at t_impulse only
        for bi in range(32):
            ti = int(out["t_impulse"][bi])
            clean_v = out["stream_a_clean"]["video"][bi]
            corr_v = out["stream_a_corrupt"]["video"][bi]
            assert np.max(np.abs(clean_v[ti] - corr_v[ti])) > 0.1
            assert np.max(np.abs(np.delete(clean_v, ti, axis=0) - np.delete(corr_v, ti, axis=0))) == 0.0


def test_gate1_decoupled_oracle_matrix_fair_16x16_rollout_advantage():
    res = run_gate1_2x2_oracle_control_matrix(seed=31001, n_worlds=128, horizon_ticks=180, rollout_horizon=14)
    arms = res["arms"]
    assert res["occupancy_grid_res"] == 16
    assert len(arms) == 9
    assert arms["oracle_perc_oracle_dyn_rollout"]["closest_approach_tail16"] >= 0.85
    assert arms["oracle_perc_oracle_dyn_rollout"]["sustained_hold_tail16"] >= 0.80
    # Oracle rollout must beat fair 16x16 static geodesic, nominal dynamics, AND exact-state reactive!
    assert arms["oracle_perc_oracle_dyn_rollout"]["sustained_hold_tail16"] > arms["oracle_perc_static_geodesic_no_rollout"]["sustained_hold_tail16"]
    assert arms["oracle_perc_oracle_dyn_rollout"]["sustained_hold_tail16"] > arms["oracle_perc_nominal_dyn_rollout"]["sustained_hold_tail16"]
    assert arms["oracle_perc_oracle_dyn_rollout"]["sustained_hold_tail16"] > arms["exact_state_reactive"]["sustained_hold_tail16"]
    assert res["paired_contrasts"]["oracle_rollout_vs_static_geodesic_hold"]["ci95_low_pp"] > 10.0
    assert res["paired_contrasts"]["oracle_rollout_vs_nominal_dyn_hold"]["ci95_low_pp"] > 10.0
    assert res["paired_contrasts"]["oracle_rollout_vs_exact_reactive_hold"]["ci95_low_pp"] > 10.0
    assert res["paired_contrasts"]["noisy_oracle_rollout_vs_noisy_static_geodesic_hold"]["delta_pp"] > 5.0

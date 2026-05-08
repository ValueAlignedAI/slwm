# SLWM-124M Sprint T1 — Text/Code Baseline Training

**Sprint:** T1 — Text/code baseline training  
**Owner role:** Training  
**Scope:** text/code only; no audio or visual data.  
**Status:** dataset/training protocol plus dependency-light tiny pilot runner. Full GPT-2-size training remains blocked by prepared external datasets and compute.

## 1. Applicable requirements

| ID / source | Requirement | T1 handling |
|---|---|---|
| T1 / `sprint_playbook_prompts.md` | Train/evaluate GPT-2-style decoder, SLWM text-only, and no-spectral SLWM text ablation. | Configs: `configs/t1/gpt2_text_tiny_pilot.json`, `configs/t1/slwm_text_tiny_pilot.json`, `configs/t1/slwm_text_no_spectral_tiny_pilot.json`. |
| T1 KPI | Report validation loss/perplexity, sample generations, throughput, memory, decoding settings. | `training/t1_text_baseline.py` writes metrics, samples, report, checkpoint, registry. |
| G-R0-1 | Quantify text/code tradeoff vs GPT-2; register tolerance before comparing. | Tiny configs register `guardrail_tolerance_percent=20.0`; full plan keeps suggested 15–25% range. |
| DD-R1-001 | GPT-2 baseline before text/code claims. | `EXP-T1-001` is the required baseline anchor. |
| DD-R1-002 | Use BPE first for comparability. | Full dataset plan uses GPT-2 BPE. Tiny local pilot uses a clearly labeled byte fallback because the repo remains dependency-light. |
| DD-R1-020 | Register every experiment. | T1 runner writes registry entries under `experiments/text/t1/<EXP-ID>/registry.json`. |
| Data contract | T1 uses only `text_code` modality id 1. | `data/text_code.py` rejects non-text modalities in T1 configs. |

## 2. Dataset decision record

### Full T1 target datasets

The proper T1 dataset path is documented in `configs/t1/text_code_full_dataset_plan.json`:

- English text: FineWeb or FineWeb-Edu curated subset.
- GPT-2 comparison reference if available: OpenWebText subset.
- Code: The Stack v2 or StarCoderData with permissive-license filters.
- Initial mixture from the research plan: 70% English text, 15% code, 10% markdown/docs, 5% math/structured text.

Required checks before a result can support G-R0-1:

1. dataset snapshot/version pinned;
2. document-level train/validation/test split manifest saved and hashed;
3. duplicate/leakage checks recorded before training;
4. code license and secrets/PII filters recorded;
5. same tokenizer, split, sequence length, optimizer family, seed set, and train-token budget used for GPT-2 and SLWM variants where feasible.

### Runnable local pilot dataset

The dependency-free T1 runner currently uses inline project-authored text/code records from config files. These are for **pipeline validation only**, not model-quality evidence. Metrics from these pilots can prove that the T1 machinery runs and logs required artifacts, but they must not be used to claim GPT-2-scale language quality.

## 3. Tokenizer decision

Full T1 evidence should use GPT-2 BPE for both GPT-2 and SLWM text-only variants, per DD-R1-002.

The tiny pilot uses `ByteFallbackTokenizer`:

- `0`: pad,
- `1`: EOS,
- `2..257`: raw UTF-8 bytes,
- `vocab_size=260` in pilot configs.

This fallback is recorded as `effective_type=byte_fallback` and `intended_tokenizer=gpt2_bpe`. It is acceptable only for local smoke/pilot artifacts.

## 4. Runnable commands

```bash
python -m training.t1_text_baseline --config configs/t1/gpt2_text_tiny_pilot.json
python -m training.t1_text_baseline --config configs/t1/slwm_text_tiny_pilot.json
python -m training.t1_text_baseline --config configs/t1/slwm_text_no_spectral_tiny_pilot.json
```

Each run writes:

- `registry.json`,
- `metrics.json`,
- `samples.json`,
- `report.md`,
- `checkpoint.npz`,
- copied `config.json`.

## 5. Acceptance criteria for T1 evidence

A T1 comparison can update the text/code guardrail only if:

1. `EXP-T1-001` GPT-2 baseline and SLWM variants complete without NaN/Inf or loss explosion;
2. all variants use the same tokenizer and train/validation split manifest;
3. validation loss/perplexity are reported for the same split;
4. sample generations include exact decoding settings and seeds;
5. throughput and memory are logged;
6. strict/core parameter accounting is recorded;
7. if SLWM underperforms, the relative loss delta is recorded rather than hidden.

## 6. Current limitations

- The committed runner is a NumPy pilot, not a large-scale GPU trainer.
- The inline corpus is too small for language-model quality claims.
- GPT-2 BPE is documented as the full-run tokenizer but not implemented in the dependency-free runner.
- No LAMBADA, HumanEval, or MBPP claim should be made until an evaluation harness and suitable trained checkpoints exist.
- No hallucination, grounding, policy, audio, or visual claims are in T1 scope.
- The current tiny SLWM core still instantiates audio/visual adapter parameters even though T1 trains only text/code data. T1 metrics keep those parameters in strict totals and label them as inactive adapter parameters; full text-only accounting should disable unused adapters or report both modes.

## 7. Current tiny pilot snapshot

The first dependency-light T1 pilot artifacts are registered under `experiments/text/t1/`.

| Experiment | Variant | Validation loss | Validation PPL | Relative text-loss delta vs `EXP-T1-001` | Guardrail status |
|---|---|---:|---:|---:|---|
| `EXP-T1-001` | GPT-2-style tiny baseline | 3.5312279394482773 | 34.16589551791503 | 0.0% | baseline anchor |
| `EXP-T1-002` | SLWM text-only tiny pilot | 5.015316550681898 | 150.7038346177376 | 42.027550661753544% | failed 20% pilot tolerance |
| `EXP-T1-003` | SLWM text-only no-spectral tiny pilot | 5.252390149267106 | 191.02229505423867 | 48.74118123588888% | failed 20% pilot tolerance |

Interpretation: the pilot proves that the T1 machinery can run and record the required artifacts. It also records the expected early text tradeoff honestly: both SLWM pilot variants underperformed the GPT-2-style pilot baseline on this tiny validation split. This result must not be generalized to full T1 until proper GPT-2 BPE datasets, scale, and compute are used.

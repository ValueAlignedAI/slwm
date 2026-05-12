# SLWM Sprint T1 — Text/Code Baseline Training

**Sprint:** T1 — Text/code baseline training  
**Owner role:** Training  
**Scope:** text/code only; no audio or visual data.  
**Status:** dataset/training protocol, dependency-light tiny pilot runner, and a PyTorch/MPS GPT-2-BPE prepared-corpus path. A limited 124M-parameter T1 benchmark has run locally; converged/full-budget GPT-2-quality training remains unachieved and must not be claimed.

## 1. Applicable requirements

| ID / source | Requirement | T1 handling |
|---|---|---|
| T1 / `docs/process/sprint_playbook_prompts.md` | Train/evaluate GPT-2-style decoder, SLWM text-only, and no-spectral SLWM text ablation. | Configs: `configs/t1/gpt2_text_tiny_pilot.json`, `configs/t1/slwm_text_tiny_pilot.json`, `configs/t1/slwm_text_no_spectral_tiny_pilot.json`. |
| T1 KPI | Report validation loss/perplexity, sample generations, throughput, memory, decoding settings. | `training/t1_text_baseline.py` writes metrics, samples, report, checkpoint, registry. |
| G-R0-1 | Quantify text/code tradeoff vs GPT-2; register tolerance before comparing. | Tiny configs register `guardrail_tolerance_percent=20.0`; full plan keeps suggested 15–25% range. |
| DD-R1-001 | GPT-2 baseline before text/code claims. | `EXP-T1-001` is the required baseline anchor. |
| DD-R1-002 | Use BPE first for comparability. | Full dataset plan uses GPT-2 BPE. Tiny local pilot uses a clearly labeled byte fallback because the repo remains dependency-light. |
| DD-R1-020 | Register every experiment. | T1 runner writes registry entries under `experiments/text/t1/<EXP-ID>/registry.json`. |
| Data contract | T1 uses only `text_code` modality id 1. | `data/text_code.py` rejects non-text modalities in T1 configs. |

Additional full-stack artifacts:

- `training/t1_prepare_text_code.py` prepares GPT-2-BPE text/code token streams, document manifests, and split hashes.
- `training/t1_torch_text.py` runs PyTorch/MPS T1 jobs from prepared corpora.
- `configs/t1/text_code_gpt2_bpe_larger_local_prepare.json` prepares the current local corpus.
- `configs/t1/gpt2_text_torch_124m_larger_local.json`, `configs/t1/slwm_text_torch_124m_larger_local.json`, and `configs/t1/slwm_text_no_spectral_torch_124m_larger_local.json` run matched 124M-scale limited-step benchmarks.

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

### Current prepared local GPT-2-BPE corpus

Prepared by:

```bash
python -m training.t1_prepare_text_code --config configs/t1/text_code_gpt2_bpe_larger_local_prepare.json
```

Prepared corpus path (git-ignored): `artifacts/t1_text_code/gpt2_bpe_larger_local_v0/`.

Prepared-corpus manifest hash:

```text
sha256:bf7f4e95cda7a7398cd02c43558e0482b266e250c579f7c4d5833c48358b077e
```

Split token hashes:

| Split | Documents | BPE tokens | Token file SHA-256 |
|---|---:|---:|---|
| train | 2249 | 1,936,862 | `sha256:2fb7cc73aa16d0746ecd65ec1a8fb1aa4de4d995c5af14a0f7d4215c7cc5d6d9` |
| validation | 20 | 15,559 | `sha256:34bc48a52b114f56a4551cd51f3cfa4c270b5bb40eefeed595da19f98ef9001b` |
| test | 31 | 26,209 | `sha256:97823c4facb470882694422807bcdb20bfdeb904746a796ee42349762f90477a` |

Sources loaded:

- `HuggingFaceFW/fineweb-edu`, config `sample-10BT` — English text component.
- `ise-uiuc/Magicoder-OSS-Instruct-75K` — accessible OSS code-like component.
- `HuggingFaceTB/smollm-corpus`, config `cosmopedia-v2` — docs/structured text component.
- `mbpp` — structured Python problem/code/test text component.

Limitation: The Stack v2 / StarCoderData were not used in this unauthenticated run because those sources are gated without a configured Hugging Face token. The current code component is therefore an accessible substitute, not the final preferred code pretraining source.

### Runnable local pilot dataset

The dependency-free T1 runner currently uses inline project-authored text/code records from config files. These are for **pipeline validation only**, not model-quality evidence. Metrics from these pilots can prove that the T1 machinery runs and logs required artifacts, but they must not be used to claim GPT-2-scale language quality.

## 3. Tokenizer decision

Full T1 evidence should use GPT-2 BPE for both GPT-2 and SLWM text-only variants, per DD-R1-002.

`data/tokenizer.py` now includes a `GPT2BPETokenizer` wrapper using Hugging Face `transformers` for the full-stack path. The tiny pilot still uses `ByteFallbackTokenizer` and remains smoke-only.

The tiny pilot uses `ByteFallbackTokenizer`:

- `0`: pad,
- `1`: EOS,
- `2..257`: raw UTF-8 bytes,
- `vocab_size=260` in pilot configs.

This fallback is recorded as `effective_type=byte_fallback` and `intended_tokenizer=gpt2_bpe`. It is acceptable only for local smoke/pilot artifacts.

## 4. Runnable commands

### GPT-2-BPE prepared-corpus 124M limited benchmark

```bash
python -m training.t1_prepare_text_code --config configs/t1/text_code_gpt2_bpe_larger_local_prepare.json
python -m training.t1_torch_text --config configs/t1/gpt2_text_torch_124m_larger_local.json
python -m training.t1_torch_text --config configs/t1/slwm_text_torch_124m_larger_local.json
python -m training.t1_torch_text --config configs/t1/slwm_text_no_spectral_torch_124m_larger_local.json
```

Each PyTorch run writes:

- `registry.json`,
- `metrics.json`,
- `samples.json`,
- `report.md`,
- `checkpoint.pt`,
- copied `config.json`.

### Dependency-light tiny pilot

```bash
python -m training.t1_text_baseline --config configs/t1/gpt2_text_tiny_pilot.json
python -m training.t1_text_baseline --config configs/t1/slwm_text_tiny_pilot.json
python -m training.t1_text_baseline --config configs/t1/slwm_text_no_spectral_tiny_pilot.json
```

Each NumPy pilot run writes:

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

- The original committed runner is a NumPy pilot; the new PyTorch/MPS runner is available for full-stack T1 mechanics.
- The inline corpus is too small for language-model quality claims.
- GPT-2 BPE is implemented for the PyTorch path, not for the dependency-light NumPy path.
- The current 124M benchmark trained only 40,960 tokens per model (20 steps × batch 1 × grad accumulation 2 × sequence 1024); this is not converged GPT-2 training.
- The current validation split is small (15,559 BPE tokens), and the benchmark configs evaluate 4,096 validation tokens (`validation_batches=4`, batch 1, sequence 1024), so loss/PPL should be treated as an initial benchmark with no confidence interval.
- Current code data is an accessible substitute because preferred The Stack v2 / StarCoderData sources were gated without authentication.
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

## 8. Current GPT-2-BPE 124M limited benchmark snapshot

The first PyTorch/MPS GPT-2-BPE benchmark artifacts are registered under `experiments/text/t1/EXP-T1-101` through `EXP-T1-103`. These use the prepared corpus above and train each 124M-scale model for the same limited token budget. They are **mechanics/initial benchmark evidence**, not converged GPT-2 quality evidence.

| Experiment | Variant | Params | Validation loss | Validation PPL | Relative text-loss delta vs `EXP-T1-101` | Throughput tokens/s | Guardrail status |
|---|---|---:|---:|---:|---:|---:|---|
| `EXP-T1-101` | GPT-2-style PyTorch/MPS baseline | 124,439,808 | 9.166085004806519 | 9567.096140493999 | 0.0% | 7913.8601317960865 | baseline anchor |
| `EXP-T1-102` | SLWM text-only PyTorch/MPS | 125,131,008 | 9.221770763397217 | 10114.95967962054 | 0.607519552366116% | 5341.088488775351 | passed 20% limited benchmark tolerance |
| `EXP-T1-103` | SLWM text-only no-spectral PyTorch/MPS | 124,826,880 | 9.140068531036377 | 9321.40391541339 | -0.28383408790665876% | 6053.176885814016 | passed 20% limited benchmark tolerance |

Interpretation: all three 124M-scale configurations train stably on the same GPT-2-BPE prepared corpus with no NaN/Inf or loss explosion. The full SLWM text-only variant is within the registered 20% T1 text guardrail in this limited benchmark; the no-spectral ablation is slightly lower loss than the GPT-2 baseline after this tiny token budget. Because training covered only 40,960 tokens per model, this should be treated as a reproducible scaling/mechanics benchmark and an initial guardrail reading, not a conclusion that SLWM matches or beats GPT-2 on text.

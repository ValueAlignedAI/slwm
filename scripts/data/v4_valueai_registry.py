"""SLWM-v4 Public Dataset Registry & License-Compliant HuggingFace Dataset Card Publisher (`valueai-org`).

Single source of truth mapping every SLWM-v4 corpus to its dedicated, license-pure public HuggingFace Dataset
repository under `valueai-org`, preserving exact upstream licenses, attributions, BibTeX citations, privacy
notices, codec specifications, and disjoint-split contracts.
"""

from __future__ import annotations

import argparse
from typing import Any

VALUEAI_ORG = "valueai-org"

VALUEAI_DATASET_CATALOG: dict[str, dict[str, Any]] = {
    "unitree_g1_h1_humanoid": {
        "repo_id": f"{VALUEAI_ORG}/slwm-v4-unitree-g1-h1-latents",
        "title": "SLWM-v4 Unitree G1/H1 Humanoid Pre-Encoded Multimodal Latents",
        "license": "apache-2.0",
        "license_display": "Apache License 2.0",
        "commercial_allowed": True,
        "upstream_repos": [
            "unitreerobotics/G1_Dex1_Assembling_Motor",
            "unitreerobotics/G1_Dex1_Assemble_Stool",
            "unitreerobotics/G1_Dex1_Hang_Pants",
            "unitreerobotics/G1_Dex1_Fold_Clothes",
            "unitreerobotics/G1_WBT_Inspire_Walk_To_Table_Put_Cups_And_Rack_Plates_In_Dishwasher",
            "unitreerobotics/G1_Brainco_Make_Sandwich",
            "unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine",
            "unitreerobotics/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher",
            "unitreerobotics/G1_Dex1_MountCamera_Dataset",
            "unitreerobotics/G1_Dex3_ToastedBread_Dataset",
            "unitreerobotics/G1_Dex1_Arrange_Surgical_Kits",
            "lerobot/unitreeh1_fold_clothes",
        ],
        "upstream_license_note": "Derived from 12 Unitree Robotics open-source G1/H1 LeRobot v3.0 datasets released under Apache-2.0 (`G1_Dex1_Wipe_Table` was omitted under the 40 h Tier-1 cap). Aligned strictly via LeRobot v3.0 `meta/episodes/` video timestamps with strictly causal zero-order hold (`0 ms` lookahead) 25 Hz kinematics; Unitree H1 (`action[40]`, `observation.state[19]`) preserves all 40 action DoFs and marks only the 19 measured body proprio joints active (zero fabrication).",
        "split_boundary": "Repository + session block (`g1h1::{repo}::epblock_{idx // 20}` for G1, `epblock_{idx // 5}` for H1), deterministic 80/10/10 train/dev/test (realized 218 train / 29 dev / 13 test clusters across WBT, Inspire, Brainco, Dex3, Dex1, and H1; 36 upstream-duplicated eval recordings excised and logged in `gate2_eval_exclusion_manifest.json`).",
        "modalities": ["video (Wan 2.1 Causal VAE int8)", "motor_action (52-DoF non-aliased)", "proprio (68-dim non-aliased + IMU quat)", "text (task UTF-8)"],
        "citation": "Unitree Robotics Open-Source Humanoid Manipulation & Whole-Body Teleoperation Datasets (2025-2026).",
    },
    "droid_1_0_1": {
        "repo_id": f"{VALUEAI_ORG}/slwm-v4-droid-latents",
        "title": "SLWM-v4 DROID 1.0.1 Pre-Encoded Multimodal Latents",
        "license": "apache-2.0",
        "license_display": "Apache License 2.0 (upstream DROID: CC-BY-4.0 / LeRobot mirror: Apache-2.0)",
        "commercial_allowed": True,
        "upstream_repos": ["lerobot/droid_1.0.1"],
        "upstream_license_note": "Derived from `lerobot/droid_1.0.1` (Apache-2.0) and the DROID dataset (Khazatsky et al., RSS 2024, CC-BY-4.0). Attribution to DROID authors is preserved.",
        "split_boundary": "Strictly building + collector disjoint (`droid::{building}::{collector_id}`), deterministic 90/5/5 train/dev/test.",
        "modalities": ["video (Wan 2.1 Causal VAE int8, 15->25 fps with explicit duplicate-group target_valid_mask)", "motor_action (8-DoF Franka in 52-DoF layout)", "proprio (68-dim layout)", "text (language instruction UTF-8)", "episode_success"],
        "citation": """@inproceedings{khazatsky2024droid,
  title={DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset},
  author={Khazatsky, Alexander and Pertsch, Karl and Nair, Suraj and others},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2024}
}""",
    },
    "agibot_and_aloha_sim": {
        "repo_id": f"{VALUEAI_ORG}/slwm-v4-agibot-aloha-latents",
        "title": "SLWM-v4 AgiBotWorld2026 & ALOHA-Sim Pre-Encoded Multimodal Latents",
        "license": "cc-by-nc-sa-4.0",
        "license_display": "CC BY-NC-SA 4.0 (Non-Commercial, ShareAlike — strictly honoring AgiBotWorld2026)",
        "commercial_allowed": False,
        "upstream_repos": [
            "agibot-world/AgiBotWorld2026",
            "lerobot/aloha_sim_transfer_cube_human",
            "lerobot/aloha_sim_insertion_human",
        ],
        "upstream_license_note": "Combines `agibot-world/AgiBotWorld2026` (licensed under **CC BY-NC-SA 4.0**, non-commercial ShareAlike) and `lerobot/aloha_sim_*` (MIT). Because AgiBotWorld2026 requires Non-Commercial and ShareAlike, this entire combined derivative dataset is released strictly under **CC BY-NC-SA 4.0** for non-commercial research use only. Note: upstream AgiBotWorld2026 `PhysicalReasoning` `succeed` is empirically 100% False (`0/3,058` positive) and is stored strictly as `-1` (unlabeled); verified dual-wrist 6D wrenches are extracted from `state[:, 2:8]` (left) and `state[:, 8:14]` (right) (`state[:, 2:14] == state[:, 14:26]`).",
        "split_boundary": "Task + scene disjoint for AgiBotWorld2026 (`agibot::task_{id}::scene_{id}`, deterministic 68/17/15 train/dev/test -> 19/4/5 clusters); 5-episode block disjoint for ALOHA-Sim (15/2/3 blocks across all 100 episodes).",
        "modalities": ["video (Wan 2.1 Causal VAE int8)", "motor_action (52-DoF non-aliased, strictly causal ZOH)", "proprio + dual-wrist 6D wrench + base IMU quat (68-dim, strictly causal ZOH)", "text (task UTF-8)"],
        "citation": "AgiBot World 2026 Team (CC BY-NC-SA 4.0) & Zhao et al., Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (RSS 2023, ALOHA).",
    },
    "rh20t_contact_av_force": {
        "repo_id": f"{VALUEAI_ORG}/slwm-v4-rh20t-latents",
        "title": "SLWM-v4 RH20T Quad-Modal (Video + Contact Audio + 6D Force/Torque + Joints) Latents",
        "license": "cc-by-nc-sa-4.0",
        "license_display": "CC BY-NC-SA 4.0 (Non-Commercial, ShareAlike — combining RH20T-C CC-BY-SA-4.0 and RH20T-NC CC-BY-NC-4.0)",
        "commercial_allowed": False,
        "upstream_repos": ["tonyfanggalaxies/RH20T"],
        "upstream_license_note": "Derived from RH20T (Fang et al., ICRA 2024). Upstream RH20T comprises `RH20T-C` (`scene_0001`–`scene_0005`, CC BY-SA 4.0) and `RH20T-NC` (`scene_0006`–`scene_0010`, CC BY-NC 4.0). To strictly respect both subsets' ShareAlike and Non-Commercial conditions, this combined latent dataset is released under **CC BY-NC-SA 4.0** for **non-commercial research only**. Per upstream RH20T privacy guidance, these compressed latents are intended solely for model training and evaluation.",
        "split_boundary": "Strictly user + scene disjoint (`rh20t::user_{user}::scene_{scene}`), deterministic 90/5/5 train/dev/test.",
        "modalities": ["video (Wan 2.1 Causal VAE int8, `fps=25:round=down`, duplicate-group masked)", "audio (Kyutai Mimi 16-RVQ 12.5 Hz, SNR-gated)", "motor_action (causal 1-step kinematic target q[t+1] in 52-DoF layout, final boundary tick masked)", "proprio + 6D right-wrist force/torque (q[t] + 6D FT in 68-dim layout)"],
        "citation": """@inproceedings{fang2024rh20t,
  title={RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot},
  author={Fang, Hao-Shu and Fang, Hongjie and Tang, Zhenyu and Liu, Jirong and Wang, Chenxi and Wang, Junbo and Zhu, Haoyi and Lu, Cewu},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  pages={653--660},
  year={2024}
}""",
    },
    "vggsound_full_sync": {
        "repo_id": f"{VALUEAI_ORG}/slwm-v4-vggsound-latents",
        "title": "SLWM-v4 VGGSound Synchronized Audio-Visual Pre-Encoded Latents",
        "license": "cc-by-4.0",
        "license_display": "CC BY 4.0 (Oxford VGGSound License)",
        "commercial_allowed": True,
        "upstream_repos": ["Loie/VGGSound"],
        "upstream_license_note": "Derived neural latent features (Wan 2.1 Causal VAE int8 + Kyutai Mimi 16-RVQ) computed from VGGSound (Chen et al., ICASSP 2020), released under **CC BY 4.0**. Copyright of original underlying YouTube videos remains with their respective creators; no raw MP4 or PCM waveforms are redistributed here.",
        "split_boundary": "Canonical YouTube ID disjoint (`yt::{youtube_id}`), shared identically with AudioCaps to prevent any cross-corpus train/eval YouTube overlap.",
        "modalities": ["video (Wan 2.1 Causal VAE int8 @ 25 fps)", "audio (Kyutai Mimi 16-RVQ @ 12.5 Hz)", "text (sound class caption UTF-8)"],
        "citation": """@inproceedings{chen2020vggsound,
  title={VGGSound: A Large-Scale Audio-Visual Dataset},
  author={Chen, Honglie and Xie, Weidi and Vedaldi, Andrea and Zisserman, Andrew},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2020}
}""",
    },
    "epic_kitchens_100_long_ego_av": {
        "repo_id": f"{VALUEAI_ORG}/slwm-v4-epic-kitchens-100-latents",
        "title": "SLWM-v4 EPIC-KITCHENS-100 Long-Horizon Egocentric AV + Causal Narration Latents",
        "license": "cc-by-nc-4.0",
        "license_display": "CC BY-NC 4.0 (Non-Commercial Research Only)",
        "commercial_allowed": False,
        "upstream_repos": [
            "a1raman/epic_kitchens_100",
            "https://github.com/epic-kitchens/epic-kitchens-100-annotations",
        ],
        "upstream_license_note": "Derived from EPIC-KITCHENS-100 (Damen et al., IJCV 2022), licensed strictly under **CC BY-NC 4.0** for **non-commercial research purposes only**. Commercial use is strictly prohibited without obtaining a separate license from the University of Bristol.",
        "split_boundary": "Strictly participant disjoint (`epic100::P01..P37`), continuous 136 s (`2.25 min`) causal segments.",
        "modalities": ["video (Wan 2.1 Causal VAE int8 @ 25 fps)", "audio (Kyutai Mimi 16-RVQ @ 12.5 Hz)", "text (official EPIC-100 narrations aligned strictly causally at stop_timestamp)"],
        "citation": """@article{damen2022rescaling,
  title={Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100},
  author={Damen, Dima and Doughty, Hazel and Farinella, Giovanni Maria and others},
  journal={International Journal of Computer Vision (IJCV)},
  volume={130},
  pages={33--55},
  year={2022}
}""",
    },
    "holoassist_and_greatesthits_av": {
        "repo_id": f"{VALUEAI_ORG}/slwm-v4-holoassist-latents",
        "title": "SLWM-v4 HoloAssist Long-Horizon Egocentric AV Latents",
        "license": "cdla-permissive-2.0",
        "license_display": "CDLA-Permissive-2.0 (Microsoft Research HoloAssist Official License)",
        "commercial_allowed": True,
        "upstream_repos": ["SitongGong/HoloAssist"],
        "upstream_license_note": "Derived from Microsoft Research HoloAssist (Wang et al., ICCV 2023), released officially under **CDLA-Permissive-2.0** (HF mirror `SitongGong/HoloAssist` tagged Apache-2.0). Encoded from the official privacy-anonymized synchronized audio-visual streams (`pitchshift_videos/*_Video_pitchshift.mp4`, `896×504` H.264 + `48 kHz` AAC). Full attribution to the Microsoft Research & ETH Zurich HoloAssist authors is preserved.",
        "split_boundary": "Strictly recording session disjoint (`holo::{session_prefix}`), continuous 136 s (`2.25 min`) causal segments.",
        "modalities": ["video (Wan 2.1 Causal VAE int8 @ 25 fps from 896×504 H.264)", "audio (Kyutai Mimi 16-RVQ @ 12.5 Hz from privacy-anonymized 48 kHz coaching+manipulation audio)"],
        "citation": """@inproceedings{wang2023holoassist,
  title={HoloAssist: An Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World},
  author={Wang, Xin and Kwon, Taein and Rad, Mahdi and Pan, Bowen and Chakraborty, Ishani and Andrist, Sean and Bohus, Dan and Feniello, Ashley and Tekin, Bugra and Frujeri, Felipe Vieira and Joshi, Neel and Pollefeys, Marc},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}""",
    },
    "greatesthits_impact_av": {
        "repo_id": f"{VALUEAI_ORG}/slwm-v4-greatesthits-latents",
        "title": "SLWM-v4 MIT CSAIL GreatestHits Impact Audio-Visual Latents",
        "license": "other",
        "license_name": "mit-csail-greatesthits-academic-noncommercial",
        "license_display": "Academic / Non-Commercial Research Only (MIT CSAIL Greatest Hits / Visually Indicated Sounds)",
        "commercial_allowed": False,
        "upstream_repos": ["https://andrewowens.com/vis/"],
        "upstream_license_note": "Derived neural latents computed from the MIT CSAIL *Visually Indicated Sounds (Greatest Hits)* dataset (Owens et al., CVPR 2016). Provided strictly for **academic and non-commercial research** with full attribution to the original authors.",
        "split_boundary": "Strictly recording-date disjoint (`gh::{YYYY-MM-DD}`), zero train/dev/test date overlap.",
        "modalities": ["video (Wan 2.1 Causal VAE int8)", "audio (Kyutai Mimi 16-RVQ @ 12.5 Hz)"],
        "citation": """@inproceedings{owens2016visually,
  title={Visually Indicated Sounds},
  author={Owens, Andrew and Isola, Phillip and McDermott, Josh and Torralba, Antonio and Adelson, Edward H. and Freeman, William T.},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016}
}""",
    },
    "wavcaps_and_libriheavy_speech": {
        "repo_id": f"{VALUEAI_ORG}/slwm-v4-speech-audio-latents",
        "title": "SLWM-v4 AudioCaps + LibriHeavy Speech & Environmental Audio Mimi 16-RVQ Latents",
        "license": "cc-by-nc-4.0",
        "license_display": "CC BY-NC 4.0 (strictly respecting OpenSound/AudioCaps CC-BY-NC-4.0 alongside LibriHeavy Apache-2.0/CC-BY-4.0)",
        "commercial_allowed": False,
        "upstream_repos": ["OpenSound/AudioCaps", "mythicinfinity/libriheavy"],
        "upstream_license_note": "Combines `OpenSound/AudioCaps` (Kim et al., NAACL 2019, **CC BY-NC 4.0**) and `mythicinfinity/libriheavy` (Kang et al., ICASSP 2024, **Apache-2.0** / LibriVox public domain audio + CC-BY-4.0 transcripts). To strictly respect AudioCaps' Non-Commercial terms, this combined dataset is licensed under **CC BY-NC 4.0** (`subcorpus` metadata tags every episode as `audiocaps` or `libriheavy`).",
        "split_boundary": "Canonical YouTube ID disjoint (`yt::{youtube_id}`, synced with VGGSound) for AudioCaps; strictly speaker-disjoint (`libriheavy::spk_{id}`) for LibriHeavy.",
        "modalities": ["audio (Kyutai Mimi 16-RVQ @ 12.5 Hz)", "text (causal aligned transcripts / audio captions UTF-8)"],
        "citation": """@inproceedings{kang2024libriheavy,
  title={Libriheavy: A 50,000 Hours ASR Corpus with Punctuation Casing and Context},
  author={Kang, Wei and Yang, Xiaoyu and Yao, Zengwei and Kuang, Fangjun and Yang, Yifan and Guo, Liyong and Lin, Long and Povey, Daniel},
  booktitle={ICASSP},
  year={2024}
}
@inproceedings{kim2019audiocaps,
  title={AudioCaps: Generating Captions for Audios in The Wild},
  author={Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee},
  booktitle={NAACL-HLT},
  year={2019}
}""",
    },
    "fineweb_edu_and_stack_code": {
        "repo_id": f"{VALUEAI_ORG}/slwm-v4-text-code-latents",
        "title": "SLWM-v4 FineWeb-Edu + GitHub Clean Code Interleaved UTF-8 Byte Shards",
        "license": "odc-by",
        "license_display": "ODC-BY 1.0 (FineWeb-Edu) + Permissive OSS Code Licenses (`codeparrot/github-code-clean`, Apache-2.0)",
        "commercial_allowed": True,
        "upstream_repos": ["HuggingFaceTB/smollm-corpus", "codeparrot/github-code-clean"],
        "upstream_license_note": "Combines `HuggingFaceTB/smollm-corpus` (`fineweb-edu-dedup`, **ODC-BY 1.0**, ~70%) and `codeparrot/github-code-clean` (**Apache-2.0** / permissive open-source licenses: MIT, BSD, Apache-2.0, ~30%). Every document is tagged with `subcorpus` (`fineweb_edu` vs `github_code_<lang>`). Attribution to HuggingFaceTB FineWeb-Edu/SmolLM and CodeParrot contributors is preserved.",
        "split_boundary": "Document SHA-256 disjoint (`textcode::{sha256}`) for prose; strictly repository-disjoint (`code_repo::{repo_name}`) for code.",
        "modalities": ["text (lossless UTF-8 bytes packed on 160 ms heartbeat schedule)"],
        "citation": "Ben Allal et al., SmolLM-Corpus / FineWeb-Edu (HuggingFaceTB, 2024, ODC-BY 1.0) & Tunstall et al., CodeParrot GitHub-Code-Clean.",
    },
    "eval_benchmarks": {
        "repo_id": f"{VALUEAI_ORG}/slwm-v4-eval-benchmarks",
        "title": "SLWM-v4 Frozen External Evaluation Benchmarks & Pre-Locked Gate 1–2 Receipts",
        "license": "cc-by-nc-4.0",
        "license_display": "Multi-License Research Compilation (Non-Commercial overall due to ESC-50 CC-BY-NC-3.0 & Clotho Academic NC)",
        "commercial_allowed": False,
        "upstream_repos": [
            "wikitext-103 (CC-BY-SA-3.0)",
            "lambada_openai (CC-BY-4.0)",
            "openai_humaneval (MIT)",
            "mbpp (CC-BY-4.0)",
            "librispeech_asr (CC-BY-4.0)",
            "clotho (Academic Non-Commercial, Tampere University)",
            "esc50 (CC-BY-NC-3.0)",
            "SLWM RSO synthetic recall benchmark (Apache-2.0)",
        ],
        "upstream_license_note": "Contains frozen evaluation splits in SLWM-v4 format plus pre-locked `showcase_manifest_common_v4.json` and Gate 2 codec/latency receipts. Individual benchmark files retain their respective upstream licenses: WikiText-103 (CC-BY-SA-3.0), LAMBADA (CC-BY-4.0), HumanEval (MIT), MBPP (CC-BY-4.0), LibriSpeech (CC-BY-4.0), Clotho (Academic Non-Commercial), ESC-50 (CC-BY-NC-3.0), and SLWM RSO (Apache-2.0). Strictly for research evaluation.",
        "split_boundary": "Official frozen validation/test evaluation benchmarks only — zero overlap with training shards.",
        "modalities": ["text", "code", "audio (Mimi 16/32-RVQ)", "synthetic recall (RSO)", "JSON governance receipts"],
        "citation": "Please cite the respective original papers for WikiText-103, LAMBADA, HumanEval, MBPP, LibriSpeech, Clotho, and ESC-50.",
    },
}


def get_valueai_repo_for_corpus(corpus_name: str) -> str:
    if corpus_name not in VALUEAI_DATASET_CATALOG:
        raise KeyError(f"Unknown corpus {corpus_name!r}; valid: {sorted(VALUEAI_DATASET_CATALOG)}")
    return str(VALUEAI_DATASET_CATALOG[corpus_name]["repo_id"])


def render_dataset_card_markdown(corpus_name: str) -> str:
    meta = VALUEAI_DATASET_CATALOG[corpus_name]
    lic_yaml = f"license: {meta['license']}\n"
    if meta["license"] == "other" and meta.get("license_name"):
        lic_yaml += f"license_name: {meta['license_name']}\nlicense_link: https://andrewowens.com/vis/\n"
    up_lines = "\n".join(f"- `{r}`" for r in meta["upstream_repos"])
    mod_lines = "\n".join(f"- {m}" for m in meta["modalities"])
    return f"""---
{lic_yaml}task_categories:
- robotics
- audio-to-audio
- video-classification
- text-generation
tags:
- slwm-v4
- world-model
- multimodal-latents
- wan2.1-vae
- kyutai-mimi
- humanoid-robotics
- streaming-160ms
pretty_name: "{meta['title']}"
---

# {meta['title']}

Part of the **SLWM-v4 (Signal-Latent World Model v4)** open multimodal streaming latent corpus published by **ValueAI** (`valueai-org`).

## License & Usage Terms

- **Dataset License:** **{meta['license_display']}**
- **Commercial Use Permitted:** `{'Yes (subject to upstream attribution/terms)' if meta['commercial_allowed'] else 'No — Non-Commercial Research Only'}`
- **Upstream Sources:**
{up_lines}

{meta['upstream_license_note']}

> **Derivative Latent Notice:** This repository distributes pre-encoded, compressed neural latent representations (`Wan 2.1 Causal VAE` per-patch affine `int8` latents and/or `Kyutai Mimi 16-RVQ` discrete acoustic codes, non-aliased `52`-DoF action / `68`-dim proprioception+force tensors, and lossless UTF-8 bytes) rather than raw uncompressed source videos or raw waveforms. All upstream copyrights, licenses, ShareAlike clauses, Non-Commercial restrictions, and attribution obligations remain strictly in effect.

## Technical Specification (`SLWM-v4` Strictly Causal `160 ms` Heartbeat)

- **Modalities:**
{mod_lines}
- **Disjoint Split Contract (`train` / `dev` / `test`):** {meta['split_boundary']}
- **Frozen Visual Codec:** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` Causal 3D VAE (`192x320` @ `25 fps`, `1 + 4H` strictly causal framing, `15 x 1024` patches per `160 ms` heartbeat), stored with Gate-2-certified per-patch affine `int8` quantization (**`61.44 dB` latent PSNR**, **`54.03 dB` decoded RGB PSNR**, **`0.99957` `11x11` SSIM**).
- **Frozen Audio Codec:** `kyutai/mimi` Causal Streaming `16-RVQ` (`24 kHz -> 12.5 Hz`, `2` audio ticks per `160 ms` heartbeat, **`4.92%` Whisper WER ceiling** on `LibriSpeech dev-clean`).
- **Embodiment Layout:** Lossless, zero-aliasing unified `52`-DoF action + `68`-dim proprioception/force-torque/IMU representation with explicit per-episode active-channel masks (`episode_action_masks`, `episode_proprio_masks`) and zero channel fabrication.
- **Per-Shard Provenance:** Every `*.npz` shard is accompanied by a paired `*_meta.json` checkpoint recording per-episode `episode_id`, `cluster_key`, split assignment, duration, and per-modality payload SHA-256 digests for non-vacuous zero-train-eval-overlap verification.

## Citation

If you use this dataset, you **must** cite the original upstream dataset authors:

```bibtex
{meta['citation']}
```
"""


def ensure_all_valueai_public_repos(*, corpora: list[str] | None = None) -> dict[str, str]:
    """Create all public `valueai-org/slwm-v4-*` dataset repos and upload their license-compliant `README.md` cards."""
    from huggingface_hub import HfApi

    api = HfApi()
    targets = corpora or list(VALUEAI_DATASET_CATALOG.keys())
    created: dict[str, str] = {}
    for cname in targets:
        meta = VALUEAI_DATASET_CATALOG[cname]
        repo_id = meta["repo_id"]
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
        # Ensure visibility is strictly public
        try:
            api.update_repo_visibility(repo_id=repo_id, private=False, repo_type="dataset")
        except Exception:
            pass
        card_md = render_dataset_card_markdown(cname).encode("utf-8")
        api.upload_file(
            path_or_fileobj=card_md,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add license-compliant SLWM-v4 dataset card ({meta['license']})",
        )
        created[cname] = repo_id
        print(f"[VALUEAI PUBLIC OK] {repo_id} (license={meta['license']})", flush=True)
    return created


def main() -> None:
    ap = argparse.ArgumentParser(description="Create & document all public SLWM-v4 datasets under valueai-org.")
    ap.add_argument("--corpora", type=str, default="", help="Optional comma-separated subset of corpus keys.")
    args = ap.parse_args()
    subset = [x.strip() for x in args.corpora.split(",") if x.strip()] if args.corpora else None
    ensure_all_valueai_public_repos(corpora=subset)


if __name__ == "__main__":
    main()

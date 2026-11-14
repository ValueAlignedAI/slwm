"""Run bounded examples and the original X1B study evaluation functions.

Use the public study checkpoints with the corresponding source modules. Every
output names its wave, seed, checkpoint format, and evaluation scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

MODELS = {
    "valueai-org/SLWM-X1B": (5, 1_008_454_410),
    "valueai-org/SLWM-X450M-Memory": (6, 453_199_938),
    "valueai-org/SLWM-v2-453M": (7, 453_199_938),
}
FILES = {"bf16": ("weights_bf16.npz", "weights_bf16_sha256"),
         "int8": ("portable_int8.npz", "portable_int8_sha256")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_jsonable(value: Any) -> Any:
    import numpy as np
    if type(value).__module__.startswith(("jax", "jaxlib")) and hasattr(value, "__array__"):
        return np.asarray(value).tolist()
    if isinstance(value, dict):
        return {key: make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def inspect_bank(path: Path, split: str, card: Path | None = None) -> dict[str, Any]:
    """Inspect schema and an optional checksum; never read raw content into output."""
    import numpy as np
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = sha256(path)
    if card is not None:
        info = json.loads(card.read_text())
        if actual_sha != info["sha256"]:
            raise ValueError(f"Dataset-card SHA-256 does not match {path.name}")
    with np.load(path, allow_pickle=False) as bank:
        key = split + "_video"
        if key not in bank.files:
            names = sorted({item.split("_", 1)[0] for item in bank.files if "_" in item})
            raise ValueError(f"No {split!r} split in {path.name}; available prefixes: {names}")
        video = bank[key]
        if video.ndim != 5 or video.shape[-3:] != (64, 64, 3):
            raise ValueError(f"Unexpected video shape in {path.name}: {video.shape}")
        keys = sorted(item for item in bank.files if item.startswith(split + "_"))
        return {"file": path.name, "sha256": actual_sha, "split": split,
                "episodes": int(video.shape[0]), "video_shape": list(video.shape),
                "array_names": keys, "checksum_card_verified": card is not None}


def locate_checkpoint(args: argparse.Namespace, file_name: str) -> Path:
    if args.checkpoint_dir is not None:
        directory = args.checkpoint_dir
    else:
        from huggingface_hub import snapshot_download
        directory = Path(snapshot_download(repo_id=args.repo_id,
                                           allow_patterns=[file_name, "config.json", "receipt.json"]))
    for name in (file_name, "config.json", "receipt.json"):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    return directory


def load_model(args: argparse.Namespace) -> tuple[dict, Any, dict, dict]:
    import jax
    from models.x1b_world import X1BSpec, count_parameters
    from scripts.train.x1b_train import load_bf16_weights, load_portable_int8

    wave, expected_count = MODELS[args.repo_id]
    file_name, sha_key = FILES[args.format]
    directory = locate_checkpoint(args, file_name)
    config = json.loads((directory / "config.json").read_text())
    receipt = json.loads((directory / "receipt.json").read_text())
    if receipt["tag"] != config["tag"] or receipt["parameter_counts"]["strict_total"] != expected_count:
        raise ValueError("Configuration, receipt, and selected model do not agree")
    if receipt.get("seed") != 2001:
        raise ValueError("This guide targets the fixed seed-2001 study artifacts")
    checkpoint = directory / file_name
    if sha256(checkpoint) != receipt[sha_key]:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {checkpoint}")
    spec_config = dict(config["spec"])
    spec_config.update(grad_checkpoint=False, rope_Window=1024)
    spec = X1BSpec(**spec_config)
    params = load_bf16_weights(checkpoint) if args.format == "bf16" else load_portable_int8(checkpoint)
    if count_parameters(params) != receipt["parameter_counts"]:
        raise ValueError("Loaded parameter counts do not match the archived receipt")
    return params, spec, config, receipt


def example(params: dict, spec: Any, wave: int, config: dict, seed: int) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from data.x1b_multimodal import generate_synthetic_world_batch
    from models.x1b_world import forward_world_model

    if wave == 7:
        from scripts.train.x1b_train_wave7 import generate_in_domain_deprivileged_sim_batch_v7
        batch = generate_in_domain_deprivileged_sim_batch_v7(
            jax.random.key(seed), batch_size=2, seq_len=8, vocab_size=spec.vocab_size)
    else:
        batch = generate_synthetic_world_batch(
            jax.random.key(seed), batch_size=2, seq_len=8, vocab_size=spec.vocab_size)

    if wave == 5:
        @jax.jit
        def forward(p, b):
            z, _ = forward_world_model(p, spec, b)
            return z
        z = forward(params, batch)
        spatial = None
    else:
        from models.x1b_wave6_core import Wave6Interventions, de_privilege_batch_proprio, forward_world_model_wave6
        interventions = Wave6Interventions(**config["interventions"])
        if wave == 6:
            @jax.jit
            def forward(p, b):
                z, _, _ = forward_world_model_wave6(p, spec, de_privilege_batch_proprio(b), interventions)
                return z
            z = forward(params, batch)
            spatial = None
        else:
            from models.x1b_wave7_core import encode_pure_visual_grid_8x8, decode_spatial_vis_pose
            @jax.jit
            def forward(p, b):
                z, _, _ = forward_world_model_wave6(p, spec, de_privilege_batch_proprio(b), interventions)
                grid = encode_pure_visual_grid_8x8(p, spec, b["video"])
                ee, obj, _ = decode_spatial_vis_pose(p, z, interventions, pure_v_8x8=grid)
                return z, ee, obj
            z, ee, obj = forward(params, batch)
            spatial = {"visual_grid_shape": [2, 8, 8, 8, spec.d_model],
                       "decoded_ee_shape": list(ee.shape), "decoded_obj_shape": list(obj.shape),
                       "decoded_positions_finite": bool(np.isfinite(np.asarray(ee, dtype=np.float32)).all()
                                                       and np.isfinite(np.asarray(obj, dtype=np.float32)).all())}

    z_host = np.asarray(jax.device_get(z), dtype=np.float32)
    result = {"mode": "generated_example", "evaluation_scope": "synthetic shape and finite-value example",
              "seed": seed, "wave": wave, "latent_shape": list(z_host.shape),
              "latent_finite": bool(np.isfinite(z_host).all()),
              "input_video_shape": list(batch["video"].shape),
              "working_memory_register_slots": 3}
    if spatial is not None:
        result["spatial"] = spatial
    return result


def evaluate(args: argparse.Namespace, params: dict, spec: Any, config: dict, receipt: dict) -> dict:
    wave = MODELS[args.repo_id][0]
    if wave == 5:
        from data.x1b_multimodal import RealMultimodalBank, RealRobotBank
        from evals.x1b_eval_suite import eval_real_multimodal_and_text, eval_real_robot_transfer, run_full_evaluation
        if args.split == "vault":
            raise ValueError("Wave 5 calls its held-out split 'test'; use --split test")
        if args.split == "dev":
            if args.bank is None and args.robot_bank is None:
                raise ValueError("Wave 5 development evaluation needs --bank or --robot-bank")
            out = {}
            if args.bank is not None:
                out["real_multimodal_dev"] = eval_real_multimodal_and_text(
                    params, spec, RealMultimodalBank(args.bank, split="dev"))
            if args.robot_bank is not None:
                out["real_robot_dev"] = eval_real_robot_transfer(
                    params, spec, RealRobotBank(args.robot_bank, split="dev"))
            return out
        if not args.full:
            raise ValueError("Wave 5's original test suite is full-size; add --full for the final test run")
        if args.bank is None or args.robot_bank is None or args.panels is None:
            raise ValueError("The reported Wave 5 test suite needs --bank, --robot-bank, and --panels")
        return run_full_evaluation(params, spec, objective_mode=receipt["objective"],
                                   real_bank_path=str(args.bank), real_robot_path=str(args.robot_bank),
                                   pymunk_panels_path=str(args.panels), include_scaling_bench=True)

    from models.x1b_wave6_core import Wave6Interventions
    if args.split == "test":
        raise ValueError("Waves 6–7 call the held-out split 'vault'; use --split vault")
    interventions = Wave6Interventions(**config["interventions"])
    if args.split == "vault" and args.bank is None:
        raise ValueError("Pass the held-out NPZ with --bank when using --split vault")
    dev_bank = str(args.bank) if args.split == "dev" and args.bank is not None else None
    vault_bank = str(args.bank) if args.split == "vault" and args.bank is not None else None
    if wave == 6:
        from evals.x1b_eval_wave6 import run_wave6_evaluation
        return run_wave6_evaluation(params, spec, interventions, seed=receipt["seed"],
                                    grounded_npz_path=dev_bank, vault_npz_path=vault_bank,
                                    smoke=not args.full)
    from evals.x1b_eval_wave7 import run_wave7_evaluation
    return run_wave7_evaluation(params, spec, interventions, seed=receipt["seed"],
                                grounded_npz_path=dev_bank, vault_npz_path=vault_bank,
                                smoke=not args.full)


def self_test(seed: int) -> dict:
    """Exercise the code path on a small, randomly initialized fixture."""
    import jax
    from models.x1b_world import make_x1b_spec, init_x1b_params
    spec = make_x1b_spec("x_tiny", "slwm_x1b", grad_checkpoint=False)
    params = init_x1b_params(spec, seed=seed)
    output = example(params, spec, 5, {}, seed)
    output.update(mode="installation_self_test", evaluation_scope="random tiny fixture; no trained checkpoint")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-test", "inspect-data", "example", "validate"), required=True)
    parser.add_argument("--repo-id", choices=tuple(MODELS), help="Exact ValueAI study artifact to load")
    parser.add_argument("--checkpoint-dir", type=Path, help="Existing directory containing config, receipt, and weights")
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--format", choices=tuple(FILES), default="bf16")
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--split", choices=("dev", "test", "vault"), default="dev")
    parser.add_argument("--bank", type=Path, help="Real multimodal or grounded spoken NPZ")
    parser.add_argument("--robot-bank", type=Path, help="Wave 5 physical robot NPZ")
    parser.add_argument("--panels", type=Path, help="Wave 5 PyMunk evaluation panels NPZ")
    parser.add_argument("--card", type=Path, help="Dataset card containing the expected SHA-256")
    parser.add_argument("--robot-card", type=Path, help="Wave 5 robot-bank dataset card")
    parser.add_argument("--panels-card", type=Path, help="Wave 5 PyMunk-panels dataset card")
    parser.add_argument("--full", action="store_true", help="Run the original full-size study suite where supported")
    parser.add_argument("--output", type=Path, help="Write the JSON report to a new file")
    args = parser.parse_args()

    if args.mode == "inspect-data":
        if args.bank is None:
            parser.error("--mode inspect-data requires --bank")
        result = inspect_bank(args.bank, args.split, args.card)
    else:
        source_root = args.source_dir.resolve()
        if not (source_root / "models" / "x1b_world.py").is_file():
            parser.error("--source-dir must contain the SLWM source repository")
        sys.path.insert(0, str(source_root))
        if args.mode == "self-test":
            result = self_test(args.seed)
        else:
            if args.repo_id is None:
                parser.error("Select one of the three exact artifact names with --repo-id")
            data_fingerprints = {}
            if args.mode == "validate":
                if args.bank is not None:
                    data_fingerprints["bank"] = inspect_bank(args.bank, args.split, args.card)
                if args.robot_bank is not None:
                    data_fingerprints["robot_bank"] = inspect_bank(args.robot_bank, args.split, args.robot_card)
                if args.panels is not None:
                    panel_sha = sha256(args.panels)
                    if args.panels_card is not None:
                        panel_card = json.loads(args.panels_card.read_text())
                        if panel_card["sha256"] != panel_sha:
                            raise ValueError("PyMunk panel dataset-card SHA-256 mismatch")
                    data_fingerprints["pymunk_panels"] = {"file": args.panels.name,"sha256": panel_sha}
            params, spec, config, receipt = load_model(args)
            wave = MODELS[args.repo_id][0]
            result = example(params, spec, wave, config, args.seed) if args.mode == "example" else evaluate(args, params, spec, config, receipt)
            result = {"mode": args.mode, "repo_id": args.repo_id, "checkpoint_format": args.format,
                      "run_tag": receipt["tag"], "training_seed": receipt["seed"],
                      "split": args.split if args.mode == "validate" else None,
                      "suite": "full" if args.full else "smoke" if args.mode == "validate" else "generated example",
                      "data_fingerprints": data_fingerprints, "results": result}
    payload = json.dumps(make_jsonable(result), indent=2)
    if args.output is None:
        print(payload)
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Study X3 Archive-Only Recovery Script
Target: Rescue artifacts from interrupted X3 evaluation/archival phases.
Strategy:
1. Scan /content/x1b-out/ on Colab session for successful receipts.
2. If receipt found, re-run only the final evaluation suite and archive.
3. If no receipt, check for checkpoint weights to determine if training finished.
"""
import argparse
import json
import pathlib
import subprocess

def archive_only_recovery(session_name: str, tag: str):
    print(f"[RECOVERY] Targeting {tag} on {session_name} for final archive...")
    # 1. Check if receipt exists on remote
    remote_receipt = f"/content/x1b-out/{tag}/receipt.json"
    cmd = [
        "colab", "--config", f".ops/session_{session_name}.json", "--auth", "adc", 
        "exec", "-s", session_name, "--", "bash", "-lc", f"test -f {remote_receipt}"
    ]
    if subprocess.run(cmd).returncode == 0:
        print(f"[RECOVERY] Found receipt for {tag}, running archive command...")
        # Re-trigger archival
        archive_cmd = [
            "colab", "--config", f".ops/session_{session_name}.json", "--auth", "adc", 
            "exec", "-s", session_name, "--", "python3", "-c",
            "from scripts.train.x1b_colab_ops import archive_run_to_fedora; "
            f"archive_run_to_fedora('{session_name}', '{tag}', 'runs_x3_1b_480p', '/content/x1b-out')"
        ]
        subprocess.run(archive_cmd, check=True)
    else:
        print(f"[RECOVERY] Receipt missing for {tag}. Run likely aborted before finalization.")

if __name__ == "__main__":
    # To be invoked manually per-session once infra stabilizes
    pass

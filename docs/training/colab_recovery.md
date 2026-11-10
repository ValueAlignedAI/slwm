# Recovering SLWM training across Colab sessions

The Mac controls runtimes with Google's Colab CLI. Colab and Fedora upload training artifacts directly to the private Google Drive vault. Fedora also keeps an independent local copy of new checkpoint generations. A generation becomes recoverable after its files and final manifest pass checksum verification.

## Connect to a runtime

The recovery drill used Colab CLI 0.7.5, a dedicated SSH identity and a separate Google application-default login on the Mac. The SSH transport runs through Google's authenticated WebSocket service. The notebook execution API provides a second route into an existing assignment. [Colab CLI](https://github.com/googlecolab/google-colab-cli) · [SSH transport](https://github.com/googlecolab/google-colab-cli/blob/main/docs/06_ssh_access.md)

From the SLWM checkout, start a CPU assignment for the recovery test.

```sh
session=slwm-recovery-01
bash scripts/train/colab_session.sh new "$session" cpu
bash scripts/train/colab_session.sh prepare "$session"
bash scripts/train/colab_session.sh ssh "$session" hostname
```

The accelerator arguments for a separately scheduled TPU experiment are `v5e1` and `v6e1`. Choose a new name for a replacement assignment. A disconnected shell can reattach to the same name while Google still lists the assignment. For a 12-hour session, reserve time for the last checkpoint upload and check its receipt before releasing the VM. The useful recovery point after preemption is the newest verified generation.

The Mac configuration lives in `~/.config/slwm-colab`. The Colab CLI uses the `gcloud` subdirectory for application-default credentials. SSH uses `session_ed25519`. The dedicated `rclone.conf` contains the single `slwm-drive` remote with the Google Drive `drive.file` scope. On Colab, `prepare` places that transfer grant in a private memory-backed directory for the life of the runtime. The Fedora transfer account keeps its own protected configuration so uploads continue with the Mac asleep.

## Run the interruption drill

```sh
run=slwm-drill-01
bash scripts/train/colab_session.sh ssh "$session" python /content/slwm-ops/colab_recovery_drill.py produce --run "$run"
bash scripts/train/colab_session.sh stop "$session"
replacement=slwm-recovery-02
bash scripts/train/colab_session.sh new "$replacement" cpu
bash scripts/train/colab_session.sh prepare "$replacement"
bash scripts/train/colab_session.sh ssh "$replacement" python /content/slwm-ops/colab_recovery_drill.py verify --run "$run"
bash scripts/train/colab_session.sh stop "$replacement"
```

The producer queues steps 2 and 4 while continuing to train. The verifier loads the latest checkpoint into a separately initialized model and compares the next four losses, weights, Adam state and both random generators with an uninterrupted eight-step run. For execution through the notebook API, use `bash scripts/train/colab_session.sh exec "$session" --file job.py --timeout 180`.

## Queue and restore a real run

On the compute machine, run the uploader alongside the trainer. The training process publishes each full optimizer snapshot by atomic rename and then calls `capture`. Most files stage through a hard link. The source checkpoint and spool must use the same physical mount path, including when the checkout itself uses a bind mount. Small configuration files can be copied between filesystems.

```sh
python colab_artifacts.py sync --spool /content/slwm-spool --watch
```

From the training process or a second shell, stage the published state. Use distinct run names for each seed and configuration.

```sh
python colab_artifacts.py capture --spool /content/slwm-spool --run pilot-81 --step 250 \
  --file resume.pt=/content/run/resume.pt --file config.json=/content/run/config.json
python colab_artifacts.py restore --run pilot-81 --to /content/recovered-pilot-81
```

The uploader writes content-addressed binary files, checks their SHA-256 digests, then writes and verifies the generation manifest. A transfer interruption leaves the staged generation available for retry on a surviving host. Restore checks every byte and size in a temporary directory before publishing the result. It can fall back past a damaged or missing generation. Conflicting manifests for a single run and step require inspection.

The G3 observer exports the trained deltas, Adam state, exposure chain, step, seed, parent identity, configuration and training lock. Its data sequence is reconstructed from the registered seed and step. The frozen S5 parents and prepared inputs have separate manifests. A new trainer with stochastic sampling, dropout, a scaler or a scheduler must include the corresponding state in its own optimizer snapshot.

The prepared-signal package restores into a clean SLWM checkout. Its tar members retain the training loader's relative paths. The package's inventory lists every input and size. The FineWeb token corpus, CIFAR and other source supplements, and frozen weights are separate vault runs. Install matching dependencies and the recorded code revision before continuing a full experiment. The current production G3 trainer uses CUDA-specific operations. The CPU recovery drill exercises storage and replay, and TPU training still needs an XLA implementation and a measured comparison.

For a fresh compute machine, the following commands reassemble the primary GPT-2 seed-81 inputs. Run them from the matching checkout using new restore destinations. The last restore puts the real optimizer state at the path the G3 trainer reads. Its native resume check verifies the registered training lock and parent identity.

```sh
vault=/content/slwm-ops/colab_artifacts.py
python "$vault" restore --run g3-fineweb-edu --to artifacts/text/fineweb_edu_scale_v1
python "$vault" restore --run g3-data-supplements --to artifacts/joint_transfer/g3/sources
python "$vault" restore --run g3-parent-EXP-T1-231 --to experiments/text/t1/500m-20261106/EXP-T1-231
python "$vault" restore --run g3-prepared-signals-v1 --to /content/slwm-prepared
tar --zstd --extract --no-same-owner --file /content/slwm-prepared/g3-prepared.tar.zst
python "$vault" restore --run g3-gpt2-dense-81 --to experiments/multimodal/t4/joint-transfer-20261109/gpt2-dense-81
```

## Fedora storage and restart

| Location | Purpose |
| --- | --- |
| `/mnt/slwm-data/SLWM` | Prepared data and the data upload spool |
| `/mnt/slwm-checkpoints-a/SLWM` | Active experiment trees, frozen parents and checkpoint spool |
| `/mnt/slwm-checkpoints-b/SLWM` | Independently verified archive and local checkpoint recovery vault |
| `~/.local/state/slwm` | Small operational logs and verification records |

The main and G3 experiment paths are bind mounts to HDD A. Their underlying placeholders are read-only to the training processes. The registered user services verify the required volume UUIDs and pause identified SLWM producers if a volume disappears or free space falls below 16 GiB. They resume the processes they paused after free space reaches 24 GiB and the required mounts are present. The open supervisor lock and study log retain their original inodes.

After a reboot, unlock the data volumes and activate the two existing HDD logical volume groups. The private host configuration records the expected UUIDs and exact bind paths. Reattach them with the deployed script, then inspect the services.

```sh
cd ~/dev/slwm-colab-ops
python3 slwm_storage_mounts.py attach slwm-fedora-volumes-20261109.json
systemctl --user is-active slwm-storage-guard slwm-drive-capture slwm-drive-checkpoints slwm-drive-inputs
tmux attach -t slwm-storage
```

The tmux panes log storage, migration, checkpoint and verification work. User services continue after the SSH terminal disconnects. Restore from HDD B during a Drive outage with the same verifier.

```sh
python3 colab_artifacts.py restore --remote /mnt/slwm-checkpoints-b/SLWM/vault \
  --run g3-signal-sparse-81 --to /mnt/slwm-data/SLWM/restored-sparse-81
```

The installed rclone release reports that its shared Google OAuth client is being retired. A long-lived deployment should use a separately registered client. Test access to existing file-scoped artifacts or reseed a new private vault from the HDD copies and validate a restore before retiring the original grant. [rclone Drive configuration](https://rclone.org/drive/#making-your-own-client-id)

## Measurements on 9 November 2026

| Check | Observed result |
| --- | --- |
| Release the Colab producer VM and restore in a new CPU assignment | Steps 5 through 8 matched the uninterrupted run bit for bit for losses, model, optimizer and RNG. Fedora independently reproduced the same result. |
| Restore production GPT-2 seed 81 via the Colab notebook API | The 8,000-update optimizer checkpoint loaded with 69 optimizer entries, 69 delta tensors, the expected frozen parent, training lock and exposure count. |
| Restore production sparse SLWM seed 81 from HDD B alone | Step 1,250 loaded with 153 optimizer entries and 153 delta tensors. Its 182,972,619-byte checkpoint also matched the binary SHA-256 on Drive. |
| Compare the completed research archive between the independent HDDs | 14,311 files matched, totalling 49,372,412,180 logical bytes. |
| Read back the prepared multimodal archive on Fedora and verify its Drive receipt | All 5,513 members matched the loader inventory. The input occupied 38,317,710,565 bytes. Drive independently reported SHA-256 `0e235aeb14ab1cc1b41c824b77b4bc6af390731149f5cd85a5e7de9b5e3defc0` for the 30,103,057,153-byte archive. |
| Recheck the Fedora system disk and process guard | Available capacity rose from about 8 GiB to 96 GiB. Both experiment bind paths resolved to HDD A and the 3090 resumed training. |
| Targeted Fedora regression suite | 13 checks passed, including interrupted uploads, a missing or corrupted generation, an independent disk restore during a simulated cloud outage, producer selection and log/lock descriptor preservation. |

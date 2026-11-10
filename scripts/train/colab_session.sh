#!/usr/bin/env bash
# Use Google Colab's authenticated SSH transport. Only "new" requests hardware.
set -euo pipefail
umask 077
action=${1:?usage: colab_session.sh new|status|ssh|exec|prepare|stop SESSION [args]}
name=${2:?a unique session name is required}
[[ $name =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || { echo "Invalid session name" >&2; exit 2; }
shift 2
private=${SLWM_COLAB_CONFIG:-"$HOME/.config/slwm-colab"}
export CLOUDSDK_CONFIG=${CLOUDSDK_CONFIG:-"$private/gcloud"}
cli=${COLAB_CLI:-$(command -v colab || true)}
[[ -n $cli ]] || cli="$HOME/.local/bin/colab"
here=$(cd "$(dirname "$0")" && pwd)

case "$action" in
    new)
        case ${1:-cpu} in
            cpu) exec "$cli" --auth adc new -s "$name" ;;
            v5e1|v6e1) exec "$cli" --auth adc new -s "$name" --tpu "$1" ;;
            *) echo "Choose cpu, v5e1 or v6e1" >&2; exit 2 ;;
        esac ;;
    status|stop) exec "$cli" --auth adc "$action" -s "$name" ;;
    ssh|exec|prepare) ;;
    *) echo "Unknown operation: $action" >&2; exit 2 ;;
esac

# Read Google's assignment list before attaching. A missing name is a recovery event.
sessions=$("$cli" --auth adc sessions)
endpoint=$(awk -v target="[$name]" '$1 == target {print $2}' <<<"$sessions")
[[ $endpoint =~ ^[A-Za-z0-9._-]+$ ]] || { echo "No active Colab assignment for $name" >&2; exit 3; }
if [[ $action == exec ]]; then
    exec "$cli" --auth adc exec -s "$name" "$@"
fi
key="$private/session_ed25519"
[[ -r "$key" && -r "$key.pub" ]] || { echo "Missing dedicated SSH identity: $key" >&2; exit 2; }
printf -v proxy '%q ' "$cli" --auth adc ssh --proxy-mode -s "$name" --identity "$key"
transport=(ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=40 -o IdentitiesOnly=yes
           -i "$key" -o "ProxyCommand=$proxy" -o "UserKnownHostsFile=$private/known_hosts"
           -o StrictHostKeyChecking=accept-new -o "HostKeyAlias=slwm-$endpoint")
if [[ $action == ssh ]]; then
    if [[ $# == 0 ]]; then
        exec "${transport[@]}" -t root@colab-runtime 'export RCLONE_CONFIG=/dev/shm/slwm-drive/rclone.conf; cd /content && exec bash -l'
    fi
    printf -v command '%q ' "$@"
    exec "${transport[@]}" root@colab-runtime "export RCLONE_CONFIG=/dev/shm/slwm-drive/rclone.conf; cd /content && $command"
fi

# Install the uploader on the compute host. Only the separate drive.file grant
# enters this runtime, over SSH and into tmpfs. The Colab/Cloud grant stays local.
conf="${RCLONE_CONFIG:-$private/rclone.conf}"
[[ -r $conf ]] || { echo "Missing dedicated slwm-drive configuration" >&2; exit 2; }
[[ $(rclone --config "$conf" listremotes) == "slwm-drive:" ]] ||
    { echo "Use a dedicated configuration containing only slwm-drive" >&2; exit 2; }
grep -Eq '^scope = drive\.file$' "$conf" || { echo "Expected drive.file scope" >&2; exit 2; }
"${transport[@]}" root@colab-runtime 'bash -se' < "$here/colab_runtime_prepare.sh"
for file in colab_artifacts.py colab_recovery_drill.py; do
    "${transport[@]}" root@colab-runtime "umask 077; cat > /content/slwm-ops/$file.tmp && mv /content/slwm-ops/$file.tmp /content/slwm-ops/$file" < "$here/$file"
done
"${transport[@]}" root@colab-runtime 'umask 077; cat > /dev/shm/slwm-drive/rclone.conf.tmp && mv /dev/shm/slwm-drive/rclone.conf.tmp /dev/shm/slwm-drive/rclone.conf' < "$conf"

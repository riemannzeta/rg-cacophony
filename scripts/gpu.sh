#!/usr/bin/env bash
# Manage the dedicated GPU instance (rgc-gpu) from the Mac.
#   scripts/gpu.sh start|stop|status|ip|ssh [cmd...]|push|pull
# The public IP changes on every start; `start` and `ip` refresh the `gpu` git remote.
set -euo pipefail
ID=<instance-id>
REGION=us-west-2
KEY=~/<ssh-key>.pem
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

ip() { aws ec2 describe-instances --instance-ids $ID --region $REGION \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text; }

set_remote() {
  local addr; addr=$(ip)
  git -C "$REPO" remote remove gpu 2>/dev/null || true
  git -C "$REPO" remote add gpu "ubuntu@$addr:rg-cacophony.git"
  git -C "$REPO" config core.sshCommand "ssh ${SSH_OPTS[*]}"
  echo "$addr"
}

case "${1:-status}" in
  start)
    aws ec2 start-instances --instance-ids $ID --region $REGION >/dev/null
    aws ec2 wait instance-status-ok --instance-ids $ID --region $REGION
    set_remote ;;
  stop)
    aws ec2 stop-instances --instance-ids $ID --region $REGION \
      --query 'StoppingInstances[0].CurrentState.Name' --output text ;;
  status)
    aws ec2 describe-instances --instance-ids $ID --region $REGION \
      --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress]' --output text ;;
  ip) set_remote ;;
  ssh) shift; ssh "${SSH_OPTS[@]}" "ubuntu@$(ip)" "$@" ;;
  push)
    git -C "$REPO" push -q gpu main
    ssh "${SSH_OPTS[@]}" "ubuntu@$(ip)" 'cd ~/rg-cacophony && git pull -q --ff-only origin main && git log --oneline -1' ;;
  pull)  # bring results back (summaries + figures; raw arrays stay remote unless asked)
    rsync -az -e "ssh ${SSH_OPTS[*]}" --exclude raw/ "ubuntu@$(ip):rg-cacophony/results/" "$REPO/results/" ;;
  *) echo "usage: $0 start|stop|status|ip|ssh|push|pull" >&2; exit 2 ;;
esac

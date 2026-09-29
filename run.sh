#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
if ! swapon --show | grep -q .; then
  if [ -f /home/nvidia/swapfile ]; then
    echo "Swap is off. Enable with: sudo swapon /home/nvidia/swapfile"
  fi
fi
exec python -m src.main "$@"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
if ! swapon --show | grep -q .; then
  if [ -f /home/nvidia/swapfile ]; then
    echo "Swap is off. If the next command asks for a password, run:"
    echo "  sudo swapon /home/nvidia/swapfile"
  fi
fi
if command -v nvpmodel >/dev/null 2>&1; then
  MODE="$(nvpmodel -q 2>/dev/null | awk -F': ' '/NV Power Mode/{print $2}')"
  echo "nvpmodel: ${MODE:-unknown}"
  if [ "${MODE}" != "MAXN_SUPER" ]; then
    echo "For max FPS (clears 'system throttled'):"
    echo "  sudo nvpmodel -m 2 && sudo jetson_clocks"
  fi
fi
exec python -m src.main "$@"

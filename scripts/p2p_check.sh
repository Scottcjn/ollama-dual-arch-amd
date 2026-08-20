#!/usr/bin/env bash
# Measure peer-to-peer reachability + bandwidth between every GPU pair.
# Runs inside the ollama-dual container by default (that's the ROCm that matters).
# Usage: scripts/p2p_check.sh [container] [MiB] [iters]
set -euo pipefail
C="${1:-ollama-dual}"; MB="${2:-256}"; IT="${3:-10}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== host: iommu / kernel cmdline"
grep -oE 'iommu=[a-z]+|amd_iommu=[a-z]+' /proc/cmdline || echo "(no iommu= on cmdline)"
echo "== rocm-smi topology (host)"
rocm-smi --showtopo 2>/dev/null | grep -vE '^=+$' | sed -n '1,40p' || true

if docker exec "$C" which hipcc >/dev/null 2>&1; then
  echo "== building p2p_check inside $C"
  docker cp "$HERE/p2p_check.cpp" "$C":/tmp/p2p_check.cpp
  docker exec "$C" bash -c 'cd /tmp && hipcc -O2 -o p2p_check p2p_check.cpp'
  docker exec "$C" /tmp/p2p_check "$MB" "$IT"
elif docker exec "$C" test -x /opt/rocm/bin/rocm-bandwidth-test 2>/dev/null; then
  echo "== no hipcc in $C; falling back to rocm-bandwidth-test (all-pairs copy matrix)"
  docker exec "$C" /opt/rocm/bin/rocm-bandwidth-test -a
elif command -v hipcc >/dev/null; then
  echo "== no hipcc in $C; building on host ROCm instead (note: host ROCm != container ROCm)"
  hipcc -O2 -o /tmp/p2p_check "$HERE/p2p_check.cpp" && /tmp/p2p_check "$MB" "$IT"
else
  echo "no hipcc in container or host, and no rocm-bandwidth-test. Install rocm-dev in the build image or on the host." >&2
  exit 1
fi

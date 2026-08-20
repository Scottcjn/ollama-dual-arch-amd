#!/usr/bin/env bash
# Measure peer-to-peer reachability + bandwidth between every GPU pair.
# Runs inside the ollama-dual container by default (that's the ROCm that matters).
# Usage: scripts/p2p_check.sh [container] [MiB] [iters]
set -euo pipefail
C="${1:-ollama-dual}"; MB="${2:-256}"; IT="${3:-10}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== host: iommu / kernel cmdline"
grep -oE 'iommu=[a-z]+|amd_iommu=[a-z]+' /proc/cmdline || echo "(no iommu= on cmdline)"
echo "== PCIe link of each GPU's UPSTREAM port (the slot), not the card's own link"
# A card can report x16 @ 16 GT/s on its internal bridge while the slot feeding it
# sits at x4 @ 2.5 GT/s. The kernel logs exactly this at boot; surface it.
for dev in $(lspci -Dn | awk '$2=="0300:"||$2=="0302:"||$2=="0380:"{print $1}'); do
  up="$(basename "$(dirname "$(readlink -f /sys/bus/pci/devices/$dev)")")"
  printf "  %s %-40s slot-port %s: %sx @ %s (max %sx @ %s)\n" "$dev" "$(lspci -s "$dev" | sed 's/.*: //' | cut -c1-40)" "$up" \
    "$(cat /sys/bus/pci/devices/$up/current_link_width 2>/dev/null)" "$(cat /sys/bus/pci/devices/$up/current_link_speed 2>/dev/null)" \
    "$(cat /sys/bus/pci/devices/$up/max_link_width 2>/dev/null)" "$(cat /sys/bus/pci/devices/$up/max_link_speed 2>/dev/null)"
done
(journalctl -k -b --no-pager 2>/dev/null || dmesg 2>/dev/null) | grep -E 'available PCIe bandwidth, limited by' | sed 's/^/  kernel: /' | cut -c1-200
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

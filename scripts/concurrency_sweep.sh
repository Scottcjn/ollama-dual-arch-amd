#!/usr/bin/env bash
# Aggregate tok/s vs number of concurrent requests. Uses ollama's own
# eval_count / eval_duration counters, not wall-clock guesses.
# Usage: scripts/concurrency_sweep.sh [model] [max_parallel] [num_predict]
# Server must be started with OLLAMA_NUM_PARALLEL >= max_parallel.
set -euo pipefail
MODEL="${1:-qwen2.5:32b}"; MAXN="${2:-4}"; NP="${3:-128}"
URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
PROMPT="Explain, in plain language and with no lists, how a pipelined CPU differs from a scalar one."
T="$(mktemp -d)"

one() {  # $1 = index
  curl -s "$URL/api/generate" -d "$(jq -cn --arg m "$MODEL" --arg p "$PROMPT $1" --argjson n "$NP" \
      '{model:$m,prompt:$p,stream:false,options:{num_predict:$n,temperature:0.7,seed:($n+'"$1"')}}')" \
    > "$T/r$1.json"
}

echo "model=$MODEL num_predict=$NP server=$URL"
curl -s "$URL/api/generate" -d "{\"model\":\"$MODEL\",\"prompt\":\"warm\",\"stream\":false,\"options\":{\"num_predict\":8}}" >/dev/null
printf "%-9s %-11s %-14s %-12s %-10s\n" "parallel" "wall_s" "per_req_tok/s" "aggregate" "speedup"
BASE=""
for N in $(seq 1 "$MAXN"); do
  rm -f "$T"/r*.json
  t0=$(date +%s.%N)
  for i in $(seq 1 "$N"); do one "$i" & done; wait
  wall=$(echo "$(date +%s.%N) - $t0" | bc)
  if jq -e 'select(.error != null or .eval_count == null)' "$T"/r*.json >/dev/null 2>&1; then
    echo "N=$N: a request failed:"; jq -c '{error, done_reason}' "$T"/r*.json; echo "(is OLLAMA_NUM_PARALLEL >= $N, and does N x num_ctx of KV cache fit in VRAM?)"; break
  fi
  # per-request rate = eval_count / eval_duration(ns); aggregate = total tokens / wall
  per=$(jq -s '[.[] | (.eval_count / (.eval_duration/1e9))] | add/length' "$T"/r*.json)
  tot=$(jq -s '[.[] | .eval_count] | add' "$T"/r*.json)
  agg=$(echo "scale=2; $tot / $wall" | bc)
  [ -z "$BASE" ] && BASE="$agg"
  sp=$(echo "scale=2; $agg / $BASE" | bc)
  printf "%-9s %-11.1f %-14.1f %-12s %-10s\n" "$N" "$wall" "$per" "$agg" "${sp}x"
done
echo
echo "aggregate = total generated tokens / wall-clock for the batch."
echo "Where speedup stops growing is the bandwidth knee; run at that N."
rm -rf "$T"

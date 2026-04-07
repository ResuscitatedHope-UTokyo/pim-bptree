#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_UPMEM_HOME="/home/dushuai/upmem-2025.1.0-Linux-x86_64"
UPMEM_HOME="${UPMEM_HOME:-$DEFAULT_UPMEM_HOME}"
export UPMEM_HOME
export PATH="$UPMEM_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$UPMEM_HOME/lib:${LD_LIBRARY_PATH:-}"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [output_file]"
    echo "This script runs with fixed NR_TASKLETS: 1 2 4 8 16"
    exit 1
fi

OUTPUT_FILE="${1:-$REPO_ROOT/logs/$(date +%Y%m%d%H%M%S).txt}"
TASKLETS=(1 2 4 8 16)

mkdir -p "$(dirname "$OUTPUT_FILE")"

TMP_DIR="$(mktemp -d /tmp/pim_report.XXXXXX)"
SUMMARY_CSV="$TMP_DIR/summary.csv"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "t,insert,pmerge,smerge,merge,total" > "$SUMMARY_CSV"
: > "$OUTPUT_FILE"

make -C "$SCRIPT_DIR" host >/dev/null

for t in "${TASKLETS[@]}"; do
    RUN_LOG="$TMP_DIR/run_${t}.log"

    bash "$SCRIPT_DIR/build_tests.sh" test5 "$t" >"$TMP_DIR/build_${t}.log" 2>&1
    "$SCRIPT_DIR/host" "$SCRIPT_DIR/B+Tree.dpu" >"$RUN_LOG" 2>&1

    FOUND_TASKLETS=$(grep -m1 -E '^NR_TASKLETS=' "$RUN_LOG" | sed -E 's/^NR_TASKLETS=([0-9]+),.*/\1/')
    if [[ -z "$FOUND_TASKLETS" ]]; then
        echo "Failed to parse NR_TASKLETS from run log for requested $t"
        exit 1
    fi
    if [[ "$FOUND_TASKLETS" != "$t" ]]; then
        echo "Thread mismatch: expected $t, got $FOUND_TASKLETS"
        exit 1
    fi

    {
        echo "========================================="
        printf "  NR_TASKLETS = %d\n" "$t"
        echo "========================================="
        awk -v t="$t" '
            $0 ~ ("^NR_TASKLETS=" t ", Total pairs=") {cap=1}
            cap {
                if ($0 ~ /^Insert:/) exit
                print
            }
        ' "$RUN_LOG"
        echo ""
    } >> "$OUTPUT_FILE"

    TIMING_LINE=$(grep -E 'Insert:|ParallelMerge:|SerialMerge:|Merge:|Total:' "$RUN_LOG" | tail -n 1)
    if [[ -z "$TIMING_LINE" ]]; then
        echo "Failed to parse timing line for NR_TASKLETS=$t"
        exit 1
    fi

    INSERT_SEC=$(echo "$TIMING_LINE" | sed -E 's/.*Insert: ([0-9.]+) sec.*/\1/')
    PMERGE_SEC=$(echo "$TIMING_LINE" | sed -E 's/.*ParallelMerge: ([0-9.]+) sec.*/\1/')
    SMERGE_SEC=$(echo "$TIMING_LINE" | sed -E 's/.*SerialMerge: ([0-9.]+) sec.*/\1/')
    MERGE_SEC=$(echo "$TIMING_LINE" | sed -E 's/.*\| Merge: ([0-9.]+) sec \| Total:.*/\1/')
    TOTAL_SEC=$(echo "$TIMING_LINE" | sed -E 's/.*\| Total: ([0-9.]+) sec.*/\1/')

    echo "$t,$INSERT_SEC,$PMERGE_SEC,$SMERGE_SEC,$MERGE_SEC,$TOTAL_SEC" >> "$SUMMARY_CSV"
done

{
    echo "========================================="
    echo "  Timing Summary"
    echo "========================================="
    echo "+------------+-------------+----------------+--------------+-------------+-------------+"
    echo "| NR_TASKLETS| Insert(sec) | P-Merge(sec)   | S-Merge(sec) | Merge(sec)  | Total(sec)  |"
    echo "+------------+-------------+----------------+--------------+-------------+-------------+"
    awk -F, 'NR > 1 {printf("| %10d | %11.6f | %14.6f | %12.6f | %11.6f | %11.6f |\n", $1, $2, $3, $4, $5, $6)}' "$SUMMARY_CSV"
    echo "+------------+-------------+----------------+--------------+-------------+-------------+"
} >> "$OUTPUT_FILE"

echo "Generated report: $OUTPUT_FILE"

#!/bin/bash
set -e
cd "$(dirname "$0")"

UPMEM_HOME=/home/dushuai/upmem-2025.1.0-Linux-x86_64
export PATH=$UPMEM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$UPMEM_HOME/lib:$LD_LIBRARY_PATH

# Compile host once (if needed)
if [ ! -f host ]; then
    clang++ -O2 -std=c++17 -I$UPMEM_HOME/include/dpu -o host host.cpp -L$UPMEM_HOME/lib -ldpu
fi

declare -a TIMING_RESULTS

for T in 1 2 4 8 16; do
    echo "========================================="
    echo "  NR_TASKLETS = $T"
    echo "========================================="
    
    dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS=$T -Wl,--no-gc-sections -DSTACK_SIZE_DEFAULT=2560 \
        -o B+Tree.dpu dpu/B+Tree.c
    
    OUTPUT=$(./host B+Tree.dpu 2>&1)
    echo "$OUTPUT" | grep -E "^(NR_TASKLETS|\+|\| Thread|\|)"
    
    TIMING_LINE=$(echo "$OUTPUT" | grep "^Insert:")
    TIMING_RESULTS+=("$T $TIMING_LINE")
    echo ""
done

# Print timing summary table
echo "========================================="
echo "  Timing Summary"
echo "========================================="
echo "+------------+-------------+----------------+--------------+-------------+-------------+"
echo "| NR_TASKLETS| Insert(sec) | P-Merge(sec)   | S-Merge(sec) | Merge(sec)  | Total(sec)  |"
echo "+------------+-------------+----------------+--------------+-------------+-------------+"
for entry in "${TIMING_RESULTS[@]}"; do
    echo "$entry" | awk '{printf "| %10s | %11s | %14s | %12s | %11s | %11s |\n", $1, $3, $7, $11, $15, $19}'
done
echo "+------------+-------------+----------------+--------------+-------------+-------------+"

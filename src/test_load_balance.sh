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

for T in 1 2 4 8 16; do
    echo "========================================="
    echo "  NR_TASKLETS = $T"
    echo "========================================="
    
    dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS=$T -Wl,--no-gc-sections -DSTACK_SIZE_DEFAULT=2560 \
        -o B+Tree.dpu dpu/B+Tree.c
    
    ./host B+Tree.dpu 2>&1 | grep -E "^(NR_TASKLETS|\+|\| Thread|\||Insert:)"
    echo ""
done

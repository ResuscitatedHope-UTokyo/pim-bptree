#!/bin/bash

# Build script for running different test configurations
# This allows separating test execution to reduce WRAM usage

export PATH=/home/dushuai/upmem-2025.1.0-Linux-x86_64/bin:$PATH
export LD_LIBRARY_PATH=/home/dushuai/upmem-2025.1.0-Linux-x86_64/lib:$LD_LIBRARY_PATH

test_config=$1

case $test_config in
    "all")
        echo "=== Building with all tests (TEST_MASK=0x7) ==="
        dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS=16 -Wl,--no-gc-sections \
            -DSTACK_SIZE_DEFAULT=2048 -o B+Tree.dpu dpu/B+Tree.c
        echo "✓ Built B+Tree.dpu with all tests"
        ;;
    "test1-2")
        echo "=== Building with Tests 1+2 (TEST_MASK=0x3) ==="
        dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS=16 -DTEST_MASK=0x3 \
            -Wl,--no-gc-sections -DSTACK_SIZE_DEFAULT=2048 -o B+Tree.dpu dpu/B+Tree.c
        echo "✓ Built B+Tree.dpu with Tests 1+2"
        ;;
    "test3")
        echo "=== Building with Test 3 only (TEST_MASK=0x4) ==="
        dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS=16 -DTEST_MASK=0x4 \
            -Wl,--no-gc-sections -DSTACK_SIZE_DEFAULT=2048 -o B+Tree.dpu dpu/B+Tree.c
        echo "✓ Built B+Tree.dpu with Test 3 only"
        ;;
    *)
        echo "Usage: $0 {all|test1-2|test3}"
        echo ""
        echo "Options:"
        echo "  all      - Build with all tests enabled (full verification)"
        echo "  test1-2  - Build with Tests 1+2 (key count + leaf ordering)"
        echo "  test3    - Build with Test 3 only (tree structure)"
        exit 1
        ;;
esac

echo ""
echo "To run: ./host B+Tree.dpu"

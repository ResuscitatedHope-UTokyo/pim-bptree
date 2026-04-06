# B+ Tree Test Suite Documentation

## Overview

This document describes the three verification tests implemented for the DPU B+ tree:

- **Test 1**: Key Count Verification
- **Test 2**: Leaf Node Ordering Verification  
- **Test 3**: Tree Structure Verification

## Test 1: Key Count Verification

### Purpose
Verifies that all inserted keys are preserved in the final tree structure.

### What It Checks
- Counts total number of keys in the tree
- Compares with expected count (502,500 keys)
- Displays: "✓ ALL KEYS PRESERVED!" if passed

### Memory Usage
- Minimal (O(1) during tree traversal)
- Part of default build

### Result
```
✓ ALL KEYS PRESERVED! Merge successful!
Total Keys in Tree: 502500
Expected Keys: 502500
```

## Test 2: Leaf Node Ordering Verification

### Purpose
Ensures all keys within leaf nodes are sorted correctly, and ordering is maintained between adjacent leaves.

### What It Checks
1. **Within-leaf ordering**: For each leaf, keys[i] < keys[i+1] for all i
2. **Between-leaf ordering**: max(leaf_i) ≤ min(leaf_i+1) for adjacent leaves
3. Reports total leaves checked and violation counts

### Memory Usage
- Moderate (O(leaf_height) stack for tree traversal)
- Included in default and "test1-2" builds

### Result
```
========== TEST 2: LEAF NODE ORDERING ==========
Total leaves traversed: 8991
Within-leaf violations: 0
Between-leaf violations: 0
✓ ALL LEAVES PROPERLY ORDERED!
```

## Test 3: Tree Structure Verification

### Purpose
Validates the structural integrity of the B+ tree, specifically checking the leaf chain.

### What It Checks
1. **Leaf chain connectivity**: All leaves are connected via next/prev pointers
2. **Backward pointer correctness**: Each leaf's prev pointer points to the previous leaf
3. **Cycle detection**: No cycles exist in the leaf chain
4. **Chain completeness**: All expected leaves are in the chain

### Memory Usage
- Low (O(n) linear iteration, minimal stack)
- Can be run separately with -DTEST_MASK=0x4

### Result
```
========== TEST 3: TREE STRUCTURE VERIFICATION ==========
Total leaves in chain: 8991
Leaf chain broken (prev pointers): NO
Cycles detected: NO

✓ TEST 3 PASSED: Leaf chain is valid!
  - No cycles detected
  - All backward pointers correct
  - Leaf chain complete and intact
```

## Building and Running Tests

### Quick Start (All Tests)

```bash
cd src
make clean
make
./host B+Tree.dpu
```

### Running Individual Tests

#### Option 1: Using Build Script

```bash
cd src

# All tests (default)
bash build_tests.sh all
./host B+Tree.dpu

# Tests 1+2 only (minimal overhead)
bash build_tests.sh test1-2
./host B+Tree.dpu

# Test 3 only (tree structure)
bash build_tests.sh test3
./host B+Tree.dpu
```

#### Option 2: Direct Compilation

```bash
cd src

# Compile with specific TEST_MASK
# Test 1 only (0x1)
dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS=16 -DTEST_MASK=0x1 \
  -Wl,--no-gc-sections -DSTACK_SIZE_DEFAULT=2048 \
  -o B+Tree.dpu dpu/B+Tree.c

# Test 2 only (0x2)
dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS=16 -DTEST_MASK=0x2 \
  -Wl,--no-gc-sections -DSTACK_SIZE_DEFAULT=2048 \
  -o B+Tree.dpu dpu/B+Tree.c

# Tests 1+2 (0x3)
dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS=16 -DTEST_MASK=0x3 \
  -Wl,--no-gc-sections -DSTACK_SIZE_DEFAULT=2048 \
  -o B+Tree.dpu dpu/B+Tree.c

# Test 3 only (0x4)
dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS=16 -DTEST_MASK=0x4 \
  -Wl,--no-gc-sections -DSTACK_SIZE_DEFAULT=2048 \
  -o B+Tree.dpu dpu/B+Tree.c

# All tests (0x7 - default if omitted)
dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS=16 \
  -Wl,--no-gc-sections -DSTACK_SIZE_DEFAULT=2048 \
  -o B+Tree.dpu dpu/B+Tree.c
```

## TEST_MASK Values

The TEST_MASK is a bitmask that controls which tests are compiled:

| Value | Tests Enabled | Purpose |
|-------|---------------|---------|
| 0x1   | Test 1        | Key count only |
| 0x2   | Test 2        | Leaf ordering only |
| 0x3   | Tests 1+2     | Most common verification |
| 0x4   | Test 3        | Tree structure validation |
| 0x7   | Tests 1-3     | Full verification (default) |

## Performance Characteristics

| Test | Memory Usage | Time | Purpose |
|------|--------------|------|---------|
| Test 1 | Minimal | < 1ms | Correctness check |
| Test 2 | Moderate | ~10ms | Correctness check |
| Test 3 | Low | ~5ms | Structural verification |
| All tests | High | ~40ms total | Complete validation |

## Integration with Build System

The modular test approach allows:

1. **CI/CD pipelines**: Run different tests separately to handle resource constraints
2. **Development**: Debug specific aspects without full overhead
3. **Simulation**: Avoid stack overflow on simulators
4. **Hardware**: Run full tests on actual DPU hardware

## Expected Output Summary

```
✓ Test 1 (Key Count):        502,500 keys preserved
✓ Test 2 (Leaf Ordering):    8,990 leaf pairs checked, 0 violations
✓ Test 3 (Tree Structure):   8,991 leaves verified, no cycles, chain intact
```

## Troubleshooting

### Stack Overflow in Simulator
- Solution: Use smaller TEST_MASK to disable unused tests
- Example: `bin/build_tests.sh test3` instead of running all tests

### Missing Test Output
- Check TEST_MASK value (might be 0 which disables all tests)
- Verify compilation succeeded with `make clean && make`

### Inconsistent Results
- Clear build artifacts: `make clean`
- Recompile with specific TEST_MASK
- Check DPU device availability

## Code References

- **Tree validation**: src/dpu/B+Tree.c (lines ~1000-1100)
- **Test data structures**: src/common.h
- **Host integration**: src/host.cpp
- **Build configuration**: src/build_tests.sh, src/Makefile

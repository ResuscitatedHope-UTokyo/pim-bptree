#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  int key, value;
} kvpair_t;

typedef struct {
  uint64_t global_root;
  int global_root_size;
  int tree_height;
  uint32_t total_key_count;
  uint32_t leaf_count;
} TreeExportInfo;

#define MAX_QUERIES 1000000

// Test 2: Leaf node ordering verification
typedef struct {
  uint32_t total_leaf_pairs_checked;  // Total adjacent leaf pairs checked
  uint32_t within_leaf_violations;     // Keys not sorted within a leaf
  uint32_t between_leaf_violations;    // Max(prev_leaf) >= Min(next_leaf)
  int first_violation_key;             // First violating key found
  uint8_t violation_type;              // 1=within leaf, 2=between leaves
} LeafOrderVerification;

#endif /* _COMMON_H_ */

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

// Test 3: Tree structure completeness verification
typedef struct {
  uint32_t total_nodes_visited;        // Total nodes in tree
  uint32_t total_leaves_scanned;       // Leaves reached via chain
  uint32_t cycle_detected;             // 1 if cycle found, 0 otherwise
  uint32_t orphan_nodes;               // Nodes unreachable from root
  uint32_t broken_chain;               // 0 if leaf chain is complete
  uint32_t tree_connected;             // 1 if all nodes connected from root
} TreeStructureVerification;

// Test 4: Leaf node occupancy verification (B+ tree fill factor)
typedef struct {
  uint32_t total_leaves_checked;       // Total leaves scanned
  uint32_t min_occupancy_leaves;       // Leaves below minimum occupancy
  uint32_t max_occupancy_leaves;       // Leaves exceeding maximum occupancy
  int first_violation_leaf_size;       // Size of first violating leaf
  int min_allowed_keys;                // Minimum allowed keys per leaf (except root)
  int max_allowed_keys;                // Maximum allowed keys per leaf
  uint32_t root_leaf_checked;          // 1 if leaf is also root
} LeafOccupancyVerification;

#endif /* _COMMON_H_ */

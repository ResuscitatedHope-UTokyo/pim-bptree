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

#endif /* _COMMON_H_ */

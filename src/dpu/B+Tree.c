#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <perfcounter.h>
#include <defs.h>
#include <mram.h>
#include <string.h>
#include <barrier.h>
#include "../common.h"

__mram_noinit_keep uint64_t nr_queries;
__mram_noinit_keep kvpair_t query_buffer[MAX_QUERIES];

__mram_ptr uint8_t *mram_heap_start = __sys_used_mram_end;

BARRIER_INIT(init_barrier, NR_TASKLETS);

// --- DPU B+ Tree Implementation ---

#define MAX_KEYS 30
#define MAX_BPTREE_HEIGHT 8
// Alignment for Pointer Tagging (7 bits, 128-byte alignment)
#define NODE_ALIGNMENT 128 // 7 bits for size (0-127)
#define SIZE_MASK 0x7F     //   000001111111
#define PTR_MASK  (~0x7F) // ...111110000000

#define MAX_SUBTREES 32
#define MAX_TOTAL_ROOTS (NR_TASKLETS * MAX_SUBTREES)
#define BATCH_SIZE 16
typedef int KeyType;
typedef int ValueType;
typedef enum { LEAF_NODE, INTERNAL_NODE } NodeType;

typedef struct LeafNode LeafNode;
typedef struct InternalNode InternalNode;

// [Occupancy Embedding]
// A NodeLink stores both the pointer (offset) and the number of keys
typedef uint32_t NodeLink;

#define PACK_LINK(addr, size)   ((NodeLink)(((uint32_t)(uintptr_t)(addr) & PTR_MASK) | ((size) & SIZE_MASK)))
#define UNPACK_ADDR(link)       ((__mram_ptr void*)((uintptr_t)((link) & PTR_MASK)))
#define UNPACK_SIZE(link)       ((int)((link) & SIZE_MASK))



struct __dma_aligned LeafNode {
    NodeType type; // 4
    // int num_keys;  <-- REMOVED
    KeyType keys[MAX_KEYS]; // 4 * 30 = 120
    ValueType values[MAX_KEYS]; // 4 * 30 = 120
    NodeLink next; // 4 (Embedded size of next leaf)
    __mram_ptr struct LeafNode *prev;// 4
};

struct __dma_aligned InternalNode {
    NodeType type; // 4
    // int num_keys;  <-- REMOVED
    KeyType keys[MAX_KEYS]; // 120
    NodeLink children[MAX_KEYS + 1]; // 4 * 31 = 124
};

typedef union __dma_aligned {
    LeafNode leaf;
    InternalNode internal;
    uint8_t raw[256];
} Node;

typedef struct {
    __mram_ptr void* root;
    __mram_ptr uint8_t* mram_ptr;
    __mram_ptr uint8_t* mram_end;
    __dma_aligned kvpair_t local_queries[BATCH_SIZE];
    Node node_buf;
} ThreadLocalData;

// --- Parallel ---

typedef struct {
    __mram_ptr void* root;
    int root_size; 
    KeyType min_key;
    KeyType max_key;
    int height;
} Subtree;

typedef struct {
    Subtree subtrees[MAX_SUBTREES];
    int count;
} ThreadForest;

ThreadLocalData thread_data[NR_TASKLETS];
ThreadForest thread_forests[NR_TASKLETS];
KeyType split_keys[NR_TASKLETS + 1];
__mram_ptr void* global_root = NULL;
int global_root_size_var = 0;

__mram_ptr void *mram_alloc_local(ThreadLocalData *tls, size_t n)
{
    // Align current pointer first
    uintptr_t current = (uintptr_t)tls->mram_ptr;
    uintptr_t offset = current & (NODE_ALIGNMENT - 1);
    if (offset != 0) {
        tls->mram_ptr += (NODE_ALIGNMENT - offset);
    }

    // [Occupancy Embedding]
    // Force 128-byte alignment so low 7 bits are 0 for pointer tagging
    size_t size = (n + (NODE_ALIGNMENT - 1)) & ~(NODE_ALIGNMENT - 1);// 上取整为128的倍数
    if (tls->mram_ptr + size > tls->mram_end) {
        return NULL; 
    }
    __mram_ptr void* p = tls->mram_ptr;
    tls->mram_ptr += size;
    return p;
}

__mram_ptr LeafNode* create_leaf_node(ThreadLocalData *tls) {
    LeafNode node __dma_aligned;
    node.type = LEAF_NODE;
    // node.num_keys = 0;  <-- REMOVED (size is stored in parent)
    node.next = PACK_LINK(NULL, 0);
    node.prev = NULL;
    
    __mram_ptr LeafNode *mram_node = (__mram_ptr LeafNode *)mram_alloc_local(tls, sizeof(LeafNode));
    if (mram_node) mram_write(&node, mram_node, sizeof(LeafNode));
    return mram_node;
}

__mram_ptr InternalNode* create_internal_node(ThreadLocalData *tls) {
    InternalNode node __dma_aligned;
    node.type = INTERNAL_NODE;
    // node.num_keys = 0; <-- REMOVED

    __mram_ptr InternalNode *mram_node = (__mram_ptr InternalNode *)mram_alloc_local(tls, sizeof(InternalNode));
    if (mram_node) mram_write(&node, mram_node, sizeof(InternalNode));
    return mram_node;
}


static void __attribute__((noinline)) split_leaf_node(ThreadLocalData *tls, __mram_ptr LeafNode *leaf_addr, LeafNode *leaf_wram, int leaf_current_size, KeyType key, ValueType value, KeyType *split_key, __mram_ptr LeafNode **new_leaf_out, int *new_leaf_size_out, int *old_leaf_new_size_out) {
    LeafNode new_leaf_wram __dma_aligned;
    __mram_ptr LeafNode *new_leaf_addr = create_leaf_node(tls);
    int total_keys = leaf_current_size + 1;
    int split_val = total_keys / 2;
    int pos = 0;
    while (pos < leaf_current_size && leaf_wram->keys[pos] < key) pos++;
    
    new_leaf_wram.type = LEAF_NODE;
    new_leaf_wram.next = leaf_wram->next;
    new_leaf_wram.prev = leaf_addr;

    int right_count = total_keys - split_val;
    
    if (pos >= split_val) {
        // New key goes to Right Node
        int r_idx = 0;
        for (int k = split_val; k < pos; k++) {
            new_leaf_wram.keys[r_idx] = leaf_wram->keys[k];
            new_leaf_wram.values[r_idx] = leaf_wram->values[k];
            r_idx++;
        }
        // Insert new key
        new_leaf_wram.keys[r_idx] = key;
        new_leaf_wram.values[r_idx] = value;
        r_idx++;
        for (int k = pos; k < leaf_current_size; k++) {
            new_leaf_wram.keys[r_idx] = leaf_wram->keys[k];
            new_leaf_wram.values[r_idx] = leaf_wram->values[k];
            r_idx++;
        }
    } else {
        // New key goes to Left Node
        for (int k = 0; k < right_count; k++) {
             new_leaf_wram.keys[k] = leaf_wram->keys[split_val - 1 + k];
             new_leaf_wram.values[k] = leaf_wram->values[split_val - 1 + k];
        }
        
        // Update Left Node
        for (int k = split_val - 1; k > pos; k--) {
            leaf_wram->keys[k] = leaf_wram->keys[k-1];
            leaf_wram->values[k] = leaf_wram->values[k-1];
        }
        leaf_wram->keys[pos] = key;
        leaf_wram->values[pos] = value;
    }
    
    int old_leaf_new_size = split_val;
    int new_leaf_size = right_count;
    
    leaf_wram->next = PACK_LINK(new_leaf_addr, new_leaf_size);
    
    __mram_ptr LeafNode* next_leaf_addr = UNPACK_ADDR(new_leaf_wram.next);

    if (next_leaf_addr != NULL) {
        LeafNode next_leaf __dma_aligned;
        mram_read(next_leaf_addr, &next_leaf, sizeof(LeafNode));
        next_leaf.prev = new_leaf_addr;
        mram_write(&next_leaf, next_leaf_addr, sizeof(LeafNode));
    }
    
    *split_key = new_leaf_wram.keys[0];
    *new_leaf_out = new_leaf_addr;
    *new_leaf_size_out = new_leaf_size;
    *old_leaf_new_size_out = old_leaf_new_size;
    
    mram_write(&new_leaf_wram, new_leaf_addr, sizeof(LeafNode));
    mram_write(leaf_wram, leaf_addr, sizeof(LeafNode));
}

static void __attribute__((noinline)) split_internal_node(ThreadLocalData *tls, __mram_ptr InternalNode *node_addr, InternalNode *node_wram, int node_current_size, KeyType key, NodeLink child_link, KeyType *up_key, __mram_ptr InternalNode **new_node_out, int *new_node_size_out, int *old_node_new_size_out) {
    InternalNode new_node_wram __dma_aligned;
    __mram_ptr InternalNode *new_node_addr = create_internal_node(tls);
    
    int total_keys = node_current_size + 1; 
    int split_idx = (total_keys) / 2;
    int pos = 0;
    while (pos < node_current_size && node_wram->keys[pos] < key) pos++;
    
    new_node_wram.type = INTERNAL_NODE;

    if (pos == split_idx) {
        *up_key = key;
        new_node_wram.children[0] = child_link;
        int count = 0;
        for (int k = pos; k < node_current_size; k++) {
            new_node_wram.keys[count] = node_wram->keys[k];
            new_node_wram.children[count+1] = node_wram->children[k+1];
            count++;
        }
        *new_node_size_out = count;
        *old_node_new_size_out = pos;
        
    } else if (pos > split_idx) {
        *up_key = node_wram->keys[split_idx];
        
        int r_idx = 0;
        // Right starts with child at split_idx + 1
        new_node_wram.children[0] = node_wram->children[split_idx + 1];
        
        // Copy keys from split_idx+1 to pos
        for (int k = split_idx + 1; k < pos; k++) {
            new_node_wram.keys[r_idx] = node_wram->keys[k];
            new_node_wram.children[r_idx+1] = node_wram->children[k+1];
            r_idx++;
        }
        
        new_node_wram.keys[r_idx] = key;
        new_node_wram.children[r_idx+1] = child_link;
        r_idx++;

        for (int k = pos; k < node_current_size; k++) {
            new_node_wram.keys[r_idx] = node_wram->keys[k];
            new_node_wram.children[r_idx+1] = node_wram->children[k+1];
            r_idx++;
        }
        
        *new_node_size_out = r_idx;
        *old_node_new_size_out = split_idx;
        
    } else { // pos < split_idx
        *up_key = node_wram->keys[split_idx - 1]; 
        
        new_node_wram.children[0] = node_wram->children[split_idx];
        int r_idx = 0;
        for (int k = split_idx; k < node_current_size; k++) {
            new_node_wram.keys[r_idx] = node_wram->keys[k];
            new_node_wram.children[r_idx+1] = node_wram->children[k+1];
            r_idx++;
        }
        *new_node_size_out = r_idx;

        for (int k = split_idx - 1; k > pos; k--) {
             node_wram->keys[k] = node_wram->keys[k-1];
             node_wram->children[k+1] = node_wram->children[k];
        }
        node_wram->keys[pos] = key;
        node_wram->children[pos+1] = child_link;
        
        *old_node_new_size_out = split_idx;
    }
    
    *new_node_out = new_node_addr;
    mram_write(&new_node_wram, new_node_addr, sizeof(InternalNode));
    mram_write(node_wram, node_addr, sizeof(InternalNode));
}

static int find_child_index(InternalNode *internal, int num_keys, KeyType key) {
    for (int i = 0; i < num_keys; i++) {
        if (internal->keys[i] > key) return i;
    }
    return num_keys;
}

// 分裂结果结构体 - 用于返回子树根分裂信息
typedef struct {
    bool did_split;           // 是否发生了根分裂
    __mram_ptr void* new_root; // 分裂产生的新子树根
    int new_root_size;        // 新子树根的大小
    KeyType split_key;        // 分裂键（新子树的最小键）
} RootSplitInfo;

bool bptree_insert_thread_ex(int thread_id, __mram_ptr void** root_ptr, int* root_size_ptr, 
                              KeyType key, ValueType value, RootSplitInfo* split_info) {
    ThreadLocalData *tls = &thread_data[thread_id];
    if (split_info) {
        split_info->did_split = false;
        split_info->new_root = NULL;
        split_info->new_root_size = 0;
    }
    
    if (*root_ptr == NULL) {
        __mram_ptr LeafNode *root = create_leaf_node(tls);
        LeafNode root_wram __dma_aligned;
        root_wram.type = LEAF_NODE;
        root_wram.keys[0] = key;
        root_wram.values[0] = value;
        root_wram.next = PACK_LINK(NULL, 0);
        root_wram.prev = NULL;
        mram_write(&root_wram, root, sizeof(LeafNode));
        *root_ptr = (__mram_ptr void*)root;
        
        *root_size_ptr = 1;
        return true;
    }
    
    __mram_ptr void* path[MAX_BPTREE_HEIGHT];
    int path_size[MAX_BPTREE_HEIGHT];
    int child_indices[MAX_BPTREE_HEIGHT];
    
    int depth = 0;
    __mram_ptr void* curr = *root_ptr;
    int curr_size = *root_size_ptr; 
    
    // [Occupancy Embedding Optimization]
    // Calculate exact transfer size based on occupancy.
    // Layout: Header(4) + Keys(120) + Children/Values...
    // Both Internal and Leaf start Children/Values at offset 124.
    // We strictly need up to 124 + (size + 1) * 4 bytes to cover:
    // - Internal: size keys, size+1 children
    // - Leaf: size keys, size values
    // Align to 8 bytes for DMA interaction.
    int read_len = (124 + (curr_size + 1) * 4 + 7) & ~7;
    if (read_len > sizeof(Node)) read_len = sizeof(Node);

    mram_read(curr, &tls->node_buf, read_len);
    while (tls->node_buf.internal.type == INTERNAL_NODE) {
        path[depth] = curr;
        path_size[depth] = curr_size;
        
        int idx = find_child_index(&tls->node_buf.internal, curr_size, key);
        child_indices[depth] = idx;
        
        depth++;
        
        // get next child
        NodeLink next_link = tls->node_buf.internal.children[idx];
        curr = UNPACK_ADDR(next_link);
        curr_size = UNPACK_SIZE(next_link);
        
        read_len = (124 + (curr_size + 1) * 4 + 7) & ~7;
        if (read_len > sizeof(Node)) read_len = sizeof(Node);
        
        mram_read(curr, &tls->node_buf, read_len);
    }

    // The optimization above reads only valid keys/values.
    // However, LeafNode stores 'next' and 'prev' pointers at the VERY END of the struct (offset 244/248).
    // If we only read partial data, 'next' and 'prev' in wram contain garbage.
    // When we later mram_write the full node, we corrupt the linked list in MRAM.
    // MUST read the full leaf node here to ensure next/prev are loaded.
    mram_read(curr, &tls->node_buf, sizeof(LeafNode));

    LeafNode *leaf = &tls->node_buf.leaf;
    // If key exists, update    
    for(int i=0; i<curr_size; i++) {
        if (leaf->keys[i] == key) {
            leaf->values[i] = value;
            mram_write(leaf, curr, sizeof(LeafNode));
            return false;
        }
    }
    
    // Insert to leaf
    if (curr_size < MAX_KEYS) {
        int pos = 0;
        while (pos < curr_size && leaf->keys[pos] < key) pos++;
        for (int k = curr_size; k > pos; k--) {
            leaf->keys[k] = leaf->keys[k-1];
            leaf->values[k] = leaf->values[k-1];
        }
        leaf->keys[pos] = key;
        leaf->values[pos] = value;
        // leaf->num_keys++; REMOVED
        
        mram_write(leaf, curr, sizeof(LeafNode));
        
        // Propagate size update up to parent
        int new_size = curr_size + 1;
        if (depth > 0) {
            __mram_ptr void* parent_ptr = path[depth-1];
            int parent_idx = child_indices[depth-1];
            InternalNode parent;
            mram_read(parent_ptr, &parent, sizeof(InternalNode));
            
            // Only update the size bits in the link
            __mram_ptr void* ptr = UNPACK_ADDR(parent.children[parent_idx]);
            parent.children[parent_idx] = PACK_LINK(ptr, new_size);
            
            mram_write(&parent, parent_ptr, sizeof(InternalNode));
        } else {
             *root_size_ptr = new_size;
        }
        return true;
    }
    
    // Split Leaf
    KeyType up_key;
    __mram_ptr LeafNode *new_leaf;
    int new_leaf_size, old_leaf_new_size;
    
    split_leaf_node(tls, (__mram_ptr LeafNode*)curr, leaf, curr_size, key, value, &up_key, &new_leaf, &new_leaf_size, &old_leaf_new_size);
    
    NodeLink child_link = PACK_LINK(new_leaf, new_leaf_size);

    if (depth > 0) {
    } else {
       *root_size_ptr = old_leaf_new_size;
    }

    // Propagate Up
    while (depth > 0) {
        depth--;
        __mram_ptr void* parent_ptr = path[depth];
        int parent_current_size = path_size[depth];
        int parent_child_idx = child_indices[depth];
        
        InternalNode parent __dma_aligned;
        mram_read(parent_ptr, &parent, sizeof(InternalNode));
        

        __mram_ptr void* old_child_ptr = UNPACK_ADDR(parent.children[parent_child_idx]);
        parent.children[parent_child_idx] = PACK_LINK(old_child_ptr, old_leaf_new_size);
        // Case 1: parent node has space
        if (parent_current_size < MAX_KEYS) {
            int pos = 0;
            while (pos < parent_current_size && parent.keys[pos] < up_key) pos++;
            for(int k=parent_current_size; k>pos; k--) {
                parent.keys[k] = parent.keys[k-1];
                parent.children[k+1] = parent.children[k];
            }
            parent.keys[pos] = up_key;
            parent.children[pos+1] = child_link;
            // parent.num_keys++; REMOVED
            
            mram_write(&parent, parent_ptr, sizeof(InternalNode));
            
            // Increased this node's size, need to update its parent
            int new_parent_size = parent_current_size + 1;
            if (depth > 0) {
                 __mram_ptr void* grand_parent_ptr = path[depth-1];
                 int grand_idx = child_indices[depth-1];
                 InternalNode grand_parent;
                 mram_read(grand_parent_ptr, &grand_parent, sizeof(InternalNode));
                 grand_parent.children[grand_idx] = PACK_LINK(parent_ptr, new_parent_size);
                 mram_write(&grand_parent, grand_parent_ptr, sizeof(InternalNode));
            } else {
                *root_size_ptr = new_parent_size;
            }
            return true;
        }
        // Case 2: parent node is full, needs to split
        __mram_ptr InternalNode *new_internal;
        int new_internal_size, old_internal_new_size;
        split_internal_node(tls, (__mram_ptr InternalNode*)parent_ptr, &parent, parent_current_size, up_key, child_link, &up_key, &new_internal, &new_internal_size, &old_internal_new_size);
        
        child_link = PACK_LINK(new_internal, new_internal_size);
        
        old_leaf_new_size = old_internal_new_size; // reusing variable name 
    }
    
    // Root Split: not creating a new root, but returning split information
    // The original root retains the left half, and right half is returned as a new subtree
    *root_size_ptr = old_leaf_new_size;  // Update original root size
    
    if (split_info) {
        split_info->did_split = true;
        split_info->new_root = UNPACK_ADDR(child_link);  // Split generated new node
        split_info->new_root_size = UNPACK_SIZE(child_link);
        split_info->split_key = up_key;
    }
    
    return true;
}

bool bptree_insert_thread(int thread_id, __mram_ptr void** root_ptr, int* root_size_ptr, KeyType key, ValueType value) {
    RootSplitInfo split_info;
    bool result = bptree_insert_thread_ex(thread_id, root_ptr, root_size_ptr, key, value, &split_info);
    
    // If root split occurred, create new root (maintain original behavior)
    if (split_info.did_split) {
        ThreadLocalData *tls = &thread_data[thread_id];
        __mram_ptr InternalNode *new_root = create_internal_node(tls);
        InternalNode new_root_wram __dma_aligned;
        memset(&new_root_wram, 0, sizeof(InternalNode));
        new_root_wram.type = INTERNAL_NODE;
        new_root_wram.children[0] = PACK_LINK(*root_ptr, *root_size_ptr);
        new_root_wram.keys[0] = split_info.split_key;
        new_root_wram.children[1] = PACK_LINK(split_info.new_root, split_info.new_root_size);
        mram_write(&new_root_wram, new_root, sizeof(InternalNode));
        *root_ptr = (__mram_ptr void*)new_root;
        *root_size_ptr = 1;
    }
    
    return result;
}

static void add_subtree(int thread_id, __mram_ptr void* node, int root_size, KeyType min_k, KeyType max_k, int height) {
    if (thread_forests[thread_id].count < MAX_SUBTREES) {
        int idx = thread_forests[thread_id].count++;
        thread_forests[thread_id].subtrees[idx].root = node;
        thread_forests[thread_id].subtrees[idx].root_size = root_size;
        thread_forests[thread_id].subtrees[idx].min_key = min_k;
        thread_forests[thread_id].subtrees[idx].max_key = max_k;
        thread_forests[thread_id].subtrees[idx].height = height;
    }
}

static void distribute_subtrees_recursive(__mram_ptr void* node, int node_size, KeyType node_min, KeyType node_max, int height) {
    if (node == NULL) return;

    // Find which thread's range fully contains this node
    int start_thread = -1;
    int end_thread = -1;
    
    for(int i=0; i<NR_TASKLETS; i++) {
        KeyType t_min = split_keys[i];
        KeyType t_max = split_keys[i+1];
        
        if (node_max > t_min && node_min < t_max) {
            if (start_thread == -1) start_thread = i;
            end_thread = i;
        }
    }
    
    if (start_thread == -1) return;
    
    // This node and its children is fully contained in ONE thread's range
    if (start_thread == end_thread) {
        KeyType t_min = split_keys[start_thread];
        KeyType t_max = split_keys[start_thread+1];
        
        if (t_min <= node_min && node_max <= t_max) {
            add_subtree(start_thread, node, node_size, node_min, node_max, height);
            return;
        }
    }
    
    NodeType type;
    // Align read to 8 bytes
    uint64_t type_buf;
    mram_read(node, &type_buf, 8);
    type = (NodeType)type_buf;
    
    // Leaf Node: Assign based on first key
    if (type == LEAF_NODE) {
        LeafNode leaf __dma_aligned;
        mram_read(node, &leaf, sizeof(LeafNode));
        KeyType start_key = (node_size > 0) ? leaf.keys[0] : node_min;
        
        for(int i=start_thread; i<=end_thread; i++) {
            if (start_key >= split_keys[i] && start_key < split_keys[i+1]) {
                add_subtree(i, node, node_size, node_min, node_max, 1);
                break;
            }
        }
        return;
    }
    
    // Internal Node: recursive
    InternalNode internal __dma_aligned;
    mram_read(node, &internal, sizeof(InternalNode));
    
    for (int i = 0; i <= node_size; i++) {
        KeyType child_min = (i == 0) ? node_min : internal.keys[i-1];
        KeyType child_max = (i == node_size) ? node_max : internal.keys[i];
        
        NodeLink child_link = internal.children[i];
        distribute_subtrees_recursive(UNPACK_ADDR(child_link), UNPACK_SIZE(child_link), child_min, child_max, height - 1);
    }
}

static void distribute_subtrees(int root_height) {
    for(int i=0; i<NR_TASKLETS; i++) {
        thread_forests[i].count = 0;
    }
    
    distribute_subtrees_recursive(global_root, global_root_size_var, INT32_MIN, INT32_MAX, root_height);
}



// Serial merge removed. New implementation coming.

// --- New Merge Algorithm ---

// Helper struct for leaf traversal
typedef struct {
    __mram_ptr LeafNode* addr;
    int size;
} LeafInfo;

LeafInfo get_leftmost_leaf(__mram_ptr void* root, int root_size, int height) {
    if (root == NULL) return (LeafInfo){NULL, 0};
    __mram_ptr void* curr = root;
    int curr_size = root_size;
    while(height > 1) {
         InternalNode node;
         mram_read(curr, &node, sizeof(InternalNode));
         NodeLink link = node.children[0];
         curr = UNPACK_ADDR(link);
         curr_size = UNPACK_SIZE(link);
         height--;
    }
    return (LeafInfo){(__mram_ptr LeafNode*)curr, curr_size};
}

LeafInfo get_rightmost_leaf(__mram_ptr void* root, int root_size, int height) {
    if (root == NULL) return (LeafInfo){NULL, 0};
    __mram_ptr void* curr = root;
    int curr_size = root_size;
    while(height > 1) {
         InternalNode node;
         mram_read(curr, &node, sizeof(InternalNode));
         // Rightmost child is at index curr_size
         NodeLink link = node.children[curr_size];
         curr = UNPACK_ADDR(link);
         curr_size = UNPACK_SIZE(link);
         height--;
    }
    return (LeafInfo){(__mram_ptr LeafNode*)curr, curr_size};
}

void connect_leaf_list(Subtree left, Subtree right) {
    if (left.root == NULL || right.root == NULL) return;
    
    LeafInfo l_info = get_rightmost_leaf(left.root, left.root_size, left.height);
    LeafInfo r_info = get_leftmost_leaf(right.root, right.root_size, right.height);
    
    if (l_info.addr && r_info.addr) {
        LeafNode l_node;
        mram_read(l_info.addr, &l_node, sizeof(LeafNode));
        
        // Link left -> right
        l_node.next = PACK_LINK(r_info.addr, r_info.size);
        mram_write(&l_node, l_info.addr, sizeof(LeafNode));
        
        LeafNode r_node;
        mram_read(r_info.addr, &r_node, sizeof(LeafNode));
        
        // Link right -> prev = left
        r_node.prev = l_info.addr;
        mram_write(&r_node, r_info.addr, sizeof(LeafNode));
    }
}

Subtree merge_same_height(int thread_id, Subtree t_left, Subtree t_right) {
    ThreadLocalData *tls = &thread_data[thread_id];
    __mram_ptr InternalNode *new_root = create_internal_node(tls);
    
    InternalNode root;
    memset(&root, 0, sizeof(InternalNode)); // Important to init children to 0/NULL
    root.type = INTERNAL_NODE;
    
    // Child 0: Left Tree
    root.children[0] = PACK_LINK(t_left.root, t_left.root_size);
    // Key 0: Right Tree's min key
    root.keys[0] = t_right.min_key;
    // Child 1: Right Tree
    root.children[1] = PACK_LINK(t_right.root, t_right.root_size);
    
    mram_write(&root, new_root, sizeof(InternalNode));
    
    Subtree res;
    res.root = (__mram_ptr void*)new_root;
    res.root_size = 1; // 1 key
    res.min_key = t_left.min_key;
    res.max_key = t_right.max_key;
    res.height = t_left.height + 1; // Height increases by 1
    return res;
}

// Helper to insert (key, link) into a node along the path and propagate splits
void insert_and_propagate(int thread_id, 
                          __mram_ptr void** path, int* path_sizes, int* child_indices, int start_depth,
                          KeyType key, NodeLink child_link,
                          Subtree* tree_to_update) {
    
    ThreadLocalData *tls = &thread_data[thread_id];
    int depth = start_depth;
    KeyType up_key = key;
    NodeLink current_child_link = child_link;
    
    while(depth >= 0) {
        __mram_ptr void* parent_ptr = path[depth];
        int parent_size = path_sizes[depth];
        
        InternalNode parent;
        mram_read(parent_ptr, &parent, sizeof(InternalNode));
        
        if (parent_size < MAX_KEYS) {
             int pos = 0;
             while (pos < parent_size && parent.keys[pos] < up_key) pos++;
             
             for(int k=parent_size; k>pos; k--) {
                parent.keys[k] = parent.keys[k-1];
                parent.children[k+1] = parent.children[k];
             }
             parent.keys[pos] = up_key;
             parent.children[pos+1] = current_child_link;
             
             mram_write(&parent, parent_ptr, sizeof(InternalNode));
             
             int current_node_new_size = parent_size + 1;
             
             // Propagate size update upwards
             for (int d = depth - 1; d >= 0; d--) {
                 InternalNode gp;
                 mram_read(path[d], &gp, sizeof(InternalNode));
                 int idx = child_indices[d];
                 gp.children[idx] = PACK_LINK(path[d+1], current_node_new_size);
                 mram_write(&gp, path[d], sizeof(InternalNode));
                 return;
             }
             
             if (depth == 0) {
                 tree_to_update->root_size = current_node_new_size;
             }
             return;
        }
        
        // Split
        __mram_ptr InternalNode *new_internal;
        int new_internal_size, new_old_size;
        split_internal_node(tls, (__mram_ptr InternalNode*)parent_ptr, &parent, parent_size, up_key, current_child_link, &up_key, &new_internal, &new_internal_size, &new_old_size);
        
        current_child_link = PACK_LINK(new_internal, new_internal_size);
        
        // Update size of the LEFT split node in the parent (grandparent)
        if (depth > 0) {
            InternalNode gp;
            mram_read(path[depth-1], &gp, sizeof(InternalNode));
            int idx = child_indices[depth-1];
            gp.children[idx] = PACK_LINK(parent_ptr, new_old_size);
            mram_write(&gp, path[depth-1], sizeof(InternalNode));
        } else {
            // Root split
             tree_to_update->root_size = new_old_size;
             
             __mram_ptr InternalNode *new_root = create_internal_node(tls);
             InternalNode r;
             memset(&r, 0, sizeof(InternalNode));
             r.type = INTERNAL_NODE;
             r.children[0] = PACK_LINK(parent_ptr, new_old_size);
             r.keys[0] = up_key;
             r.children[1] = current_child_link;
             
             mram_write(&r, new_root, sizeof(InternalNode));
             tree_to_update->root = (__mram_ptr void*)new_root;
             tree_to_update->root_size = 1;
             tree_to_update->height++;
             return; 
        }
        
        depth--;
    }
}

Subtree merge_left_shorter(int thread_id, Subtree t_left, Subtree t_right) {
    int target_h = t_left.height + 1;
    __mram_ptr void* path[MAX_BPTREE_HEIGHT];
    int path_sizes[MAX_BPTREE_HEIGHT];
    int child_indices[MAX_BPTREE_HEIGHT];
    int depth = 0;
    
    __mram_ptr void* curr = t_right.root;
    int curr_size = t_right.root_size;
    int curr_h = t_right.height;
    
    // Decend along left edge (index 0)
    while(curr_h > target_h) {
        path[depth] = curr;
        path_sizes[depth] = curr_size;
        child_indices[depth] = 0; // Always go left
        depth++;
        
        InternalNode node;
        mram_read(curr, &node, sizeof(InternalNode));
        NodeLink child = node.children[0];
        curr = UNPACK_ADDR(child);
        curr_size = UNPACK_SIZE(child);
        curr_h--;
    }
    
    path[depth] = curr;
    path_sizes[depth] = curr_size;
    
    insert_and_propagate(thread_id, path, path_sizes, child_indices, depth, 
                         t_right.min_key, PACK_LINK(t_left.root, t_left.root_size), 
                         &t_right);
                         
    t_right.min_key = t_left.min_key;
    return t_right;
}

Subtree merge_right_shorter(int thread_id, Subtree t_left, Subtree t_right) {
    int target_h = t_right.height + 1;
    __mram_ptr void* path[MAX_BPTREE_HEIGHT];
    int path_sizes[MAX_BPTREE_HEIGHT];
    int child_indices[MAX_BPTREE_HEIGHT];
    int depth = 0;
    
    __mram_ptr void* curr = t_left.root;
    int curr_size = t_left.root_size;
    int curr_h = t_left.height;
    
    // Decend along right edge
    while(curr_h > target_h) {
        path[depth] = curr;
        path_sizes[depth] = curr_size;
        child_indices[depth] = curr_size; // Go rightmost (index = size)
        depth++;
        
        InternalNode node;
        mram_read(curr, &node, sizeof(InternalNode));
        NodeLink child = node.children[curr_size];
        curr = UNPACK_ADDR(child);
        curr_size = UNPACK_SIZE(child);
        curr_h--;
    }
    
    path[depth] = curr;
    path_sizes[depth] = curr_size;
    
    insert_and_propagate(thread_id, path, path_sizes, child_indices, depth, 
                         t_right.min_key, PACK_LINK(t_right.root, t_right.root_size), 
                         &t_left);
                         
    t_left.max_key = t_right.max_key;
    return t_left;
}

Subtree merge_two_trees(int thread_id, Subtree t_left, Subtree t_right) {
    // 0. Connect Leaf Lists
    connect_leaf_list(t_left, t_right);
    
    if (t_left.height == t_right.height) {
        return merge_same_height(thread_id, t_left, t_right);
    } else if (t_left.height < t_right.height) {
        return merge_left_shorter(thread_id, t_left, t_right);
    } else {
        return merge_right_shorter(thread_id, t_left, t_right);
    }
}

void parallel_merge_local(int thread_id) {
    ThreadForest *forest = &thread_forests[thread_id];
    if (forest->count == 0) {
        // Nothing to do
    } else if (forest->count == 1) {
        // Nothing to do
    } else {
        Subtree merged = forest->subtrees[0];
        for (int j = 1; j < forest->count; j++) {
            merged = merge_two_trees(thread_id, merged, forest->subtrees[j]);
        }
        forest->subtrees[0] = merged;
        forest->count = 1;
    }
}

void serial_merge_all(int thread_id) {
    if (thread_id != 0) return;
    
    Subtree all_trees[NR_TASKLETS];
    int count = 0;
    
    for (int i=0; i<NR_TASKLETS; i++) {
        if (thread_forests[i].count > 0) {
            all_trees[count++] = thread_forests[i].subtrees[0];
        }
    }
    
    if (count == 0) {
        global_root = NULL;
        global_root_size_var = 0;
        return;
    }
    
    Subtree final_tree = all_trees[0];
    for (int i=1; i<count; i++) {
        final_tree = merge_two_trees(thread_id, final_tree, all_trees[i]);
    }
    
    global_root = final_tree.root;
    global_root_size_var = final_tree.root_size;
}

__dma_aligned kvpair_t local_queries[BATCH_SIZE];

static void serial_initialization(int thread_id) {
    // Inlined into main to save stack space
}

int main()
{
    int thread_id = me();
    

    if (thread_id >= NR_TASKLETS) return 0;
    
    uint32_t heap_size_per_thread = 3 * 1024 * 1024;// 3MB MRAM per thread
    
    thread_data[thread_id].mram_ptr = mram_heap_start + (thread_id * heap_size_per_thread);
    thread_data[thread_id].mram_end = thread_data[thread_id].mram_ptr + heap_size_per_thread;
    thread_data[thread_id].root = NULL;
    
    barrier_wait(&init_barrier);
    
    // 1. Serial Initialization (Thread 0)—— Insert 500,000 keys
    if (thread_id == 0) {
        int total_queries = 500000; 
        for (int i = 0; i < total_queries; i += BATCH_SIZE) {
            int count = total_queries - i;
            if (count > BATCH_SIZE) count = BATCH_SIZE;
            // read 64 k-v pairs per batch of queries to WRAM
            mram_read(&query_buffer[i], local_queries, count * sizeof(kvpair_t));
            
            for (int j = 0; j < count; j++) {
                bptree_insert_thread(thread_id, &global_root, &global_root_size_var, local_queries[j].key, local_queries[j].value);
            }
        }
        
        // Calculate Split Keys, traverse the leaves to find split points
        int leaf_count = 0;
        __mram_ptr void* curr = global_root;
        // Go to leftmost leaf node
        while(curr) {
            Node n;
            mram_read(curr, &n, sizeof(Node));
            if (n.internal.type == LEAF_NODE) break;
            curr = UNPACK_ADDR(n.internal.children[0]);
        }
        // calculate total leaf nodes
        __mram_ptr LeafNode* first_leaf = (__mram_ptr LeafNode*)curr;
        __mram_ptr LeafNode* l = first_leaf;
        while(l) {
            leaf_count++;
            LeafNode ln;
            mram_read(l, &ln, sizeof(LeafNode));
            l = UNPACK_ADDR(ln.next);
        }
        
        split_keys[0] = INT32_MIN;
        split_keys[NR_TASKLETS] = INT32_MAX;
        for(int i=1; i<NR_TASKLETS; i++) split_keys[i] = INT32_MAX;
        
        if (leaf_count > 0) {
            l = first_leaf;
            int current_leaf_idx = 0;
            int split_idx = 1;
            while(l && split_idx < NR_TASKLETS) {
                current_leaf_idx++;
                // split_idx-th split key at current_leaf_idx-th leaf
                int target_idx = (long)split_idx * leaf_count / NR_TASKLETS;
                
                LeafNode ln;
                mram_read(l, &ln, sizeof(LeafNode));
                
                if (current_leaf_idx == target_idx) {
                    if (UNPACK_ADDR(ln.next)) {
                        LeafNode next_ln;
                        // read next leaf
                        mram_read(UNPACK_ADDR(ln.next), &next_ln, sizeof(LeafNode));
                        // 1st key as the split key
                        split_keys[split_idx] = next_ln.keys[0];
                    }
                    split_idx++;
                }
                l = UNPACK_ADDR(ln.next);
            }
        }
        
        // 2. Distribute Subtrees
        int height = 0;
        curr = global_root;
        if (curr) {
            height = 1;
            while(1) {
                Node n;
                mram_read(curr, &n, sizeof(Node));
                if (n.internal.type == LEAF_NODE) break;
                curr = UNPACK_ADDR(n.internal.children[0]);
                height++;
            }
        }
        
        distribute_subtrees(height);
    }
    
    barrier_wait(&init_barrier);
    
    if (thread_id == 0) {
        perfcounter_config(COUNT_CYCLES, false);
    }
    
    barrier_wait(&init_barrier);
    
    int measure_start = 500000;
    int measure_count = 2500;
    int m_chunk = measure_count / NR_TASKLETS;
    int m_start = measure_start + thread_id * m_chunk;
    int m_end = measure_start + (thread_id + 1) * m_chunk;
    if (thread_id == NR_TASKLETS - 1) m_end = measure_start + measure_count;
    
    perfcounter_t start_time, end_time, insert_end_time = 0;
    if (thread_id == 0) start_time = perfcounter_get();
    

    // 并行插入 - 使用新算法：子树根分裂时不增加高度，而是产生新子树
    for (int i = m_start; i < m_end; i += BATCH_SIZE) {
        int count = m_end - i;
        if (count > BATCH_SIZE) count = BATCH_SIZE;
        // read batch of queries to WRAM
        mram_read(&query_buffer[i], thread_data[thread_id].local_queries, count * sizeof(kvpair_t));
        
        for (int j = 0; j < count; j++) {
            KeyType key = thread_data[thread_id].local_queries[j].key;
            ValueType value = thread_data[thread_id].local_queries[j].value;
            
            // Find correct subtree
            bool inserted = false;
             for(int k=0; k<thread_forests[thread_id].count; k++) {
                Subtree *s = &thread_forests[thread_id].subtrees[k];
                // Check if the key falls within the subtree's range
                if (key >= s->min_key && key < s->max_key) {
                    RootSplitInfo split_info;
                    bptree_insert_thread_ex(thread_id, &s->root, &s->root_size, key, value, &split_info);
                    
                    // 如果子树根分裂了，添加新子树到森林（高度不变）
                    if (split_info.did_split && thread_forests[thread_id].count < MAX_SUBTREES) {
                        // 更新原子树的范围（只包含左半部分）
                        KeyType old_max = s->max_key;
                        s->max_key = split_info.split_key;
                        
                        // 添加新子树（右半部分）
                        int new_idx = thread_forests[thread_id].count++;
                        thread_forests[thread_id].subtrees[new_idx].root = split_info.new_root;
                        thread_forests[thread_id].subtrees[new_idx].root_size = split_info.new_root_size;
                        thread_forests[thread_id].subtrees[new_idx].min_key = split_info.split_key;
                        thread_forests[thread_id].subtrees[new_idx].max_key = old_max;
                        thread_forests[thread_id].subtrees[new_idx].height = s->height;  // 高度不变！
                    }
                    inserted = true;
                    break; 
                } 
            }
            if (!inserted) {
                if (thread_forests[thread_id].count < MAX_SUBTREES) {
                    int idx = thread_forests[thread_id].count++;
                    thread_forests[thread_id].subtrees[idx].root = NULL;
                    thread_forests[thread_id].subtrees[idx].root_size = 0;
                    thread_forests[thread_id].subtrees[idx].min_key = key;
                    thread_forests[thread_id].subtrees[idx].max_key = key;
                    thread_forests[thread_id].subtrees[idx].height = 1;
                    bptree_insert_thread(thread_id, &thread_forests[thread_id].subtrees[idx].root, &thread_forests[thread_id].subtrees[idx].root_size, key, thread_data[thread_id].local_queries[j].value);
                }
            }
        }
    }
    
    barrier_wait(&init_barrier);
    if(thread_id == 0) insert_end_time = perfcounter_get();
    
    // Merge
    // Phase 1: Parallel Merge
    parallel_merge_local(thread_id);
    
    barrier_wait(&init_barrier);
    
    // Phase 2: Serial Merge (Thread 0)
    serial_merge_all(thread_id);

    barrier_wait(&init_barrier);
    if (thread_id == 0) end_time = perfcounter_get();   
    
    // Output (Thread 0)
    if (thread_id == 0) {
        float insert_time = (float)(insert_end_time - start_time) / (float)CLOCKS_PER_SEC;
        float merge_time = (float)(end_time - insert_end_time) / (float)CLOCKS_PER_SEC;
        float total_time = (float)(end_time - start_time) / (float)CLOCKS_PER_SEC;
        
        printf("Insert: %f sec, Merge: %f sec, Total: %f sec\n", insert_time, merge_time, total_time);
        printf("Elapsed time: %f sec\n", total_time);
       
    }
    
    return 0;
}
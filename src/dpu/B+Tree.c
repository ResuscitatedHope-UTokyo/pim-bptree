#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <perfcounter.h>
#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include "../common.h"

__mram_noinit_keep uint64_t nr_queries;
__mram_noinit_keep kvpair_t query_buffer[MAX_QUERIES];

__mram_ptr uint8_t *mram_heap_start = __sys_used_mram_end;

BARRIER_INIT(init_barrier, NR_TASKLETS);

// --- DPU B+ Tree Implementation ---

#define MAX_KEYS 30
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

KeyType first_keys[MAX_TOTAL_ROOTS];
int heights[MAX_TOTAL_ROOTS];

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
    node.next = PACK_LINK(NULL, 0); // Init next as NULL with 0 size
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

    KeyType temp_keys[MAX_KEYS + 1];
    ValueType temp_values[MAX_KEYS + 1];
    
    int pos = 0;
    while (pos < leaf_current_size && leaf_wram->keys[pos] < key) pos++;
    
    for(int k=0; k<pos; k++) {
        temp_keys[k] = leaf_wram->keys[k];
        temp_values[k] = leaf_wram->values[k];
    }
    temp_keys[pos] = key;
    temp_values[pos] = value;
    for(int k=pos; k<leaf_current_size; k++) {
        temp_keys[k+1] = leaf_wram->keys[k];
        temp_values[k+1] = leaf_wram->values[k];
    }
    
    int total = MAX_KEYS + 1;
    int split = total / 2;
    
    // Update old leaf (in WRAM)
    int old_leaf_new_size = split;
    // leaf_wram->num_keys = split; // REMOVED
    for(int k=0; k<split; k++) {
        leaf_wram->keys[k] = temp_keys[k];
        leaf_wram->values[k] = temp_values[k];
    }
    
    new_leaf_wram.type = LEAF_NODE;
    // new_leaf_wram.num_keys = total - split; // REMOVED
    int new_leaf_size = total - split;
    for(int k=0; k<new_leaf_size; k++) {
        new_leaf_wram.keys[k] = temp_keys[split + k];
        new_leaf_wram.values[k] = temp_values[split + k];
    }
    
    new_leaf_wram.next = leaf_wram->next;
    new_leaf_wram.prev = leaf_addr;
    
    // leaf_wram->next -> points to new_leaf, EMBEDDING new_leaf_size
    leaf_wram->next = PACK_LINK(new_leaf_addr, new_leaf_size);
    
    // Update next leaf's prev pointer (if exists)
    __mram_ptr LeafNode* next_leaf_addr = UNPACK_ADDR(new_leaf_wram.next);
    int next_leaf_size = UNPACK_SIZE(new_leaf_wram.next);

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
    
    KeyType temp_keys[MAX_KEYS + 1];
    NodeLink temp_children[MAX_KEYS + 2];
    
    int pos = 0;
    while (pos < node_current_size && node_wram->keys[pos] < key) pos++;
    
    for(int k=0; k<pos; k++) temp_keys[k] = node_wram->keys[k];
    for(int k=0; k<=pos; k++) temp_children[k] = node_wram->children[k];
    
    temp_keys[pos] = key;
    temp_children[pos+1] = child_link;
    
    for(int k=pos; k<node_current_size; k++) {
        temp_keys[k+1] = node_wram->keys[k];
        temp_children[k+2] = node_wram->children[k+1];
    }
    
    int total_keys = MAX_KEYS + 1;
    int split_idx = total_keys / 2;
    *up_key = temp_keys[split_idx];
    
    // left part: only save split_idx keys
    int left_size = split_idx; 
    // node_wram->num_keys = split_idx; // REMOVED
    for(int k=0; k<split_idx; k++) {
        node_wram->keys[k] = temp_keys[k];
        node_wram->children[k] = temp_children[k];
    }
    node_wram->children[split_idx] = temp_children[split_idx];
    
    // right part: from split_idx+1 to end
    new_node_wram.type = INTERNAL_NODE;
    int right_size = total_keys - split_idx - 1;
    // new_node_wram.num_keys = right_size; // REMOVED
    
    for(int k=0; k<right_size; k++) {
        new_node_wram.keys[k] = temp_keys[split_idx + 1 + k];
        new_node_wram.children[k] = temp_children[split_idx + 1 + k];
    }
    new_node_wram.children[right_size] = temp_children[total_keys];
    
    *new_node_out = new_node_addr;
    *new_node_size_out = right_size;
    *old_node_new_size_out = left_size;
    
    mram_write(&new_node_wram, new_node_addr, sizeof(InternalNode));
    mram_write(node_wram, node_addr, sizeof(InternalNode));
}

static int find_child_index(InternalNode *internal, int num_keys, KeyType key) {
    for (int i = 0; i < num_keys; i++) {
        if (internal->keys[i] > key) return i;
    }
    return num_keys;
}

bool bptree_insert_thread(int thread_id, __mram_ptr void** root_ptr, int* root_size_ptr, KeyType key, ValueType value) {
    ThreadLocalData *tls = &thread_data[thread_id];
    
    if (*root_ptr == NULL) {
        __mram_ptr LeafNode *root = create_leaf_node(tls);
        LeafNode root_wram __dma_aligned;
        root_wram.type = LEAF_NODE;
        root_wram.keys[0] = key;
        root_wram.values[0] = value;
        // root_wram.num_keys = 1; <-- REMOVED
        root_wram.next = PACK_LINK(NULL, 0);
        root_wram.prev = NULL;
        mram_write(&root_wram, root, sizeof(LeafNode));
        *root_ptr = (__mram_ptr void*)root;
        
        *root_size_ptr = 1; // Track root size
        return true;
    }
    
    __mram_ptr void* path[16];
    int path_size[16]; // Store the size of each node in path
    int child_indices[16]; // Index in parent's children array
    
    int depth = 0;
    __mram_ptr void* curr = *root_ptr;
    int curr_size = *root_size_ptr; // Start with known root size
    
    Node node_buf;
    
    mram_read(curr, &node_buf, sizeof(Node));
    while (node_buf.internal.type == INTERNAL_NODE) {
        path[depth] = curr;
        path_size[depth] = curr_size;
        
        int idx = find_child_index(&node_buf.internal, curr_size, key);
        child_indices[depth] = idx;
        
        depth++;
        
        // Unpack next child
        NodeLink next_link = node_buf.internal.children[idx];
        curr = UNPACK_ADDR(next_link);
        curr_size = UNPACK_SIZE(next_link); // Get size from pointer
        
        mram_read(curr, &node_buf, sizeof(Node));
    }

    LeafNode *leaf = &node_buf.leaf;
    // If key exists, update    
    for(int i=0; i<curr_size; i++) {
        if (leaf->keys[i] == key) {
            leaf->values[i] = value;
            mram_write(leaf, curr, sizeof(LeafNode));
            return false;
        }
    }
    
    // Insert into leaf
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
    
    // Update the OLD leaf's size in its parent
    // 'split_leaf_node' reduced the old leaf's size.
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
        
        // Update the size of the child just split (it became smaller)
        // Previous loop iteration gave us the old child's new size (stored in old_leaf_new_size variable logic, though variable names are reused)
        // For the first iteration (Leaf split), child at parent_child_idx is 'curr', and its size is now 'old_leaf_new_size'
        __mram_ptr void* old_child_ptr = UNPACK_ADDR(parent.children[parent_child_idx]);
        parent.children[parent_child_idx] = PACK_LINK(old_child_ptr, old_leaf_new_size);

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
        
        __mram_ptr InternalNode *new_internal;
        int new_internal_size, old_internal_new_size;
        split_internal_node(tls, (__mram_ptr InternalNode*)parent_ptr, &parent, parent_current_size, up_key, child_link, &up_key, &new_internal, &new_internal_size, &old_internal_new_size);
        
        child_link = PACK_LINK(new_internal, new_internal_size);
        
        // the current node 'parent' is now the 'old child' of the next level
        old_leaf_new_size = old_internal_new_size; // reusing variable name for logic flow
    }
    
    // Root Split
    __mram_ptr InternalNode *new_root = create_internal_node(tls);
    InternalNode new_root_wram __dma_aligned;
    new_root_wram.type = INTERNAL_NODE;
    
    // new_root_wram.children[0] = *root_ptr; // OLD
    // Must pack existing root with its new (reduced) size
    new_root_wram.children[0] = PACK_LINK(*root_ptr, old_leaf_new_size); 
    
    new_root_wram.keys[0] = up_key;
    new_root_wram.children[1] = child_link;
    
    // new_root_wram.num_keys = 1; REMOVED
    *root_size_ptr = 1;
    
    mram_write(&new_root_wram, new_root, sizeof(InternalNode));
    *root_ptr = (__mram_ptr void*)new_root;
    
    return true;
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

 __mram_ptr void* roots[MAX_TOTAL_ROOTS];
 int root_sizes[MAX_TOTAL_ROOTS]; 

static void merge_forests() {
    // 1. Collect all roots
    int total_roots = 0;
    int max_height = 0;
    
    for(int i=0; i<NR_TASKLETS; i++) {
        for(int j=0; j<thread_forests[i].count; j++) {
            if (total_roots < MAX_TOTAL_ROOTS) {
                roots[total_roots] = thread_forests[i].subtrees[j].root;
                root_sizes[total_roots] = thread_forests[i].subtrees[j].root_size;
                
                NodeType type;
                uint64_t type_buf;
                mram_read(roots[total_roots], &type_buf, 8);
                type = (NodeType)type_buf;

                if (type == LEAF_NODE) {
                    LeafNode l __dma_aligned;
                    mram_read(roots[total_roots], &l, sizeof(LeafNode));
                    first_keys[total_roots] = l.keys[0];
                } else {
                    first_keys[total_roots] = thread_forests[i].subtrees[j].min_key;
                }
                
                heights[total_roots] = thread_forests[i].subtrees[j].height;
                if (heights[total_roots] > max_height) max_height = heights[total_roots];
                total_roots++;
            }
        }
    }
    
    if (total_roots == 0) {
        global_root = NULL;
        return;
    }
    
    // 2. adjust heights to max_height
    ThreadLocalData *tls = &thread_data[0];
    
    for(int i=0; i<total_roots; i++) {
        while(heights[i] < max_height) {
            // Grow
            __mram_ptr InternalNode *new_node = create_internal_node(tls);
            InternalNode n __dma_aligned;
            n.type = INTERNAL_NODE;
            // n.num_keys = 0; removed
            n.children[0] = PACK_LINK(roots[i], root_sizes[i]);
            
            mram_write(&n, new_node, sizeof(InternalNode));
            roots[i] = (__mram_ptr void*)new_node;
            root_sizes[i] = 0;
            heights[i]++;
        }
    }
    
    // 3. Build 
    int current_count = total_roots;
    __mram_ptr void** current_level = roots;
    int* current_sizes = root_sizes;
    
    while(current_count > 1) {
        int parent_count = (current_count + MAX_KEYS) / (MAX_KEYS + 1); // MAX_KEYS+1 children per node
        
        int parent_idx = 0;
        for(int i=0; i<current_count; i += (MAX_KEYS + 1)) {
            int child_count = (MAX_KEYS + 1);
            if (child_count > current_count - i) child_count = current_count - i;
            
            __mram_ptr InternalNode *parent = create_internal_node(tls);
            InternalNode p __dma_aligned;
            p.type = INTERNAL_NODE;
            // p.num_keys = child_count - 1; // Removed
            
            for(int j=0; j<child_count; j++) {
                p.children[j] = PACK_LINK(current_level[i+j], current_sizes[i+j]);
                if (j > 0) {
                    NodeType type;
                    uint64_t type_buf;
                    mram_read(current_level[i+j], &type_buf, 8);
                    type = (NodeType)type_buf;

                    if (type == LEAF_NODE) {
                        LeafNode l __dma_aligned;
                        mram_read(current_level[i+j], &l, sizeof(LeafNode));
                        p.keys[j-1] = l.keys[0];
                    } else {
                        __mram_ptr void* curr = current_level[i+j];
                        while(1) {
                            InternalNode in __dma_aligned;
                            mram_read(curr, &in, sizeof(InternalNode));
                            if (in.type == LEAF_NODE) {
                                LeafNode l __dma_aligned;
                                mram_read(curr, &l, sizeof(LeafNode));
                                p.keys[j-1] = l.keys[0];
                                break;
                            }
                            curr = UNPACK_ADDR(in.children[0]);
                        }
                    }
                }
            }
            mram_write(&p, parent, sizeof(InternalNode));
            current_level[parent_idx] = (__mram_ptr void*)parent;
            current_sizes[parent_idx] = child_count - 1;
            parent_idx++;
        }
        current_count = parent_idx;
    }
    
    global_root = current_level[0];
    global_root_size_var = current_sizes[0];
}

__dma_aligned kvpair_t local_queries[BATCH_SIZE];

static void __attribute__((noinline)) serial_initialization(int thread_id) {
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
        serial_initialization(thread_id);
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
    
    perfcounter_t start_time, end_time;
    if (thread_id == 0) start_time = perfcounter_get();
    

    // kvpair_t local_queries[BATCH_SIZE];
    for (int i = m_start; i < m_end; i += BATCH_SIZE) {
        int count = m_end - i;
        if (count > BATCH_SIZE) count = BATCH_SIZE;
        // read batch of queries to WRAM
        mram_read(&query_buffer[i], thread_data[thread_id].local_queries, count * sizeof(kvpair_t));
        
        for (int j = 0; j < count; j++) {
            KeyType key = thread_data[thread_id].local_queries[j].key;
            
            // Find correct subtree
            bool inserted = false;
             for(int k=0; k<thread_forests[thread_id].count; k++) {
                Subtree *s = &thread_forests[thread_id].subtrees[k];
                // Check if the key falls within the subtree's range
                if (key >= s->min_key && key < s->max_key) {
                    bptree_insert_thread(thread_id, &s->root, &s->root_size, key, thread_data[thread_id].local_queries[j].value);
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
    
    // Merge (Thread 0)
    if (thread_id == 0) {
        perfcounter_t insert_end_time = perfcounter_get();
        merge_forests();
        end_time = perfcounter_get();
        
        float insert_time = (float)(insert_end_time - start_time) / (float)CLOCKS_PER_SEC;
        float merge_time = (float)(end_time - insert_end_time) / (float)CLOCKS_PER_SEC;
        float total_time = (float)(end_time - start_time) / (float)CLOCKS_PER_SEC;
        
        printf("Insert: %f sec, Merge: %f sec, Total: %f sec\n", insert_time, merge_time, total_time);
        printf("Elapsed time: %f sec\n", total_time);

       
    }
    
    return 0;
}

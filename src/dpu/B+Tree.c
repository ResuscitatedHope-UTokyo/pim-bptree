#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <defs.h>
#include <mram.h>
#include "../common.h"

__mram_noinit_keep uint64_t nr_queries;
__mram_noinit_keep kvpair_t query_buffer[MAX_QUERIES];

__mram_ptr uint8_t *mram_free = __sys_used_mram_end;
// --- DPU B+ Tree Implementation ---

#define MAX_KEYS 3

typedef int KeyType;
typedef int ValueType;
typedef enum { LEAF_NODE, INTERNAL_NODE } NodeType;
typedef struct LeafNode LeafNode;
// __dma_aligned
struct __dma_aligned LeafNode {
    NodeType type; // 4
    int num_keys;  // 4
    KeyType keys[MAX_KEYS]; // 4 * 3 = 12
    ValueType values[MAX_KEYS]; // 4 * 3 = 12
    __mram_ptr struct LeafNode *next; // 4
    __mram_ptr struct LeafNode *prev; // 4
    // 4 + 4 + 12 + 12 + 4 + 4 = 40
    // 40 mod 8 = 0
};

typedef struct __dma_aligned {
    NodeType type; // 4
    int num_keys;  // 4
    KeyType keys[MAX_KEYS]; // 4 * 3 = 12
    __mram_ptr void *children[MAX_KEYS + 1]; // 4 * 4 = 16
    // 4 + 4 + 12 + 16 = 36
    // int padding; // 4
    // 40 mod 8 = 0
} InternalNode;

struct NodeHeader{
    NodeType type;
    int num_keys;
};

__mram_ptr void *tree_root = NULL;
int tree_height = 0;

__mram_ptr void *mram_alloc(const size_t n);

NodeType get_node_type(__mram_ptr void *node) {
    if (node == NULL) return LEAF_NODE;
    // uint64_t buffer;
    struct NodeHeader buffer;
    mram_read(node, &buffer, 8);
    // Low 4 bytes is NodeType
    // High 4 bytes is num_keys
    // return (NodeType)(buffer & 0xFFFFFFFF);
    return buffer.type;
}

__mram_ptr LeafNode* create_leaf_node(void) {
    LeafNode node __dma_aligned;
    node.type = LEAF_NODE;
    node.num_keys = 0;
    node.next = NULL;
    node.prev = NULL;
    
    __mram_ptr LeafNode *mram_node = (__mram_ptr LeafNode *)mram_alloc(sizeof(LeafNode));
    mram_write(&node, mram_node, sizeof(LeafNode));
    return mram_node;
}

__mram_ptr InternalNode* create_internal_node(void) {
    InternalNode node __dma_aligned;
    node.type = INTERNAL_NODE;
    node.num_keys = 0;

    __mram_ptr InternalNode *mram_node = (__mram_ptr InternalNode *)mram_alloc(sizeof(InternalNode));
    mram_write(&node, mram_node, sizeof(InternalNode));
    return mram_node;
}

static void split_leaf_node(__mram_ptr LeafNode *leaf_addr, LeafNode *leaf_wram, KeyType key, ValueType value, KeyType *split_key, __mram_ptr LeafNode **new_leaf_out) {
    printf("!!! Splitting Leaf Node !!!\n");
    LeafNode new_leaf_wram __dma_aligned;
    __mram_ptr LeafNode *new_leaf_addr = create_leaf_node();

    KeyType temp_keys[MAX_KEYS + 1];
    ValueType temp_values[MAX_KEYS + 1];
    
    int pos = 0;
    while (pos < leaf_wram->num_keys && leaf_wram->keys[pos] < key) pos++;
    
    for(int k=0; k<pos; k++) {
        temp_keys[k] = leaf_wram->keys[k];
        temp_values[k] = leaf_wram->values[k];
    }
    temp_keys[pos] = key;
    temp_values[pos] = value;
    for(int k=pos; k<leaf_wram->num_keys; k++) {
        temp_keys[k+1] = leaf_wram->keys[k];
        temp_values[k+1] = leaf_wram->values[k];
    }
    
    int total = MAX_KEYS + 1;
    int split = total / 2;
    
    leaf_wram->num_keys = split;
    for(int k=0; k<split; k++) {
        leaf_wram->keys[k] = temp_keys[k];
        leaf_wram->values[k] = temp_values[k];
    }
    
    new_leaf_wram.type = LEAF_NODE;
    new_leaf_wram.num_keys = total - split;
    for(int k=0; k<new_leaf_wram.num_keys; k++) {
        new_leaf_wram.keys[k] = temp_keys[split + k];
        new_leaf_wram.values[k] = temp_values[split + k];
    }
    
    new_leaf_wram.next = leaf_wram->next;
    new_leaf_wram.prev = leaf_addr;
    leaf_wram->next = new_leaf_addr;
    
    if (new_leaf_wram.next != NULL) {
        LeafNode next_leaf __dma_aligned;
        mram_read(new_leaf_wram.next, &next_leaf, sizeof(LeafNode));
        next_leaf.prev = new_leaf_addr;
        mram_write(&next_leaf, new_leaf_wram.next, sizeof(LeafNode));
    }
    
    *split_key = new_leaf_wram.keys[0];
    *new_leaf_out = new_leaf_addr;
    
    // Write back new leaf
    mram_write(&new_leaf_wram, new_leaf_addr, sizeof(LeafNode));
    mram_write(leaf_wram, leaf_addr, sizeof(LeafNode));
}

static void split_internal_node(__mram_ptr InternalNode *node_addr, InternalNode *node_wram, KeyType key, __mram_ptr void *child, KeyType *up_key, __mram_ptr InternalNode **new_node_out) {
    printf("!!! Splitting Internal Node !!!\n");
    InternalNode new_node_wram __dma_aligned;
    __mram_ptr InternalNode *new_node_addr = create_internal_node();
    
    KeyType temp_keys[MAX_KEYS + 1];
    __mram_ptr void *temp_children[MAX_KEYS + 2];
    
    int pos = 0;
    while (pos < node_wram->num_keys && node_wram->keys[pos] < key) pos++;
    
    for(int k=0; k<pos; k++) temp_keys[k] = node_wram->keys[k];
    for(int k=0; k<=pos; k++) temp_children[k] = node_wram->children[k];
    
    temp_keys[pos] = key;
    temp_children[pos+1] = child;
    
    for(int k=pos; k<node_wram->num_keys; k++) {
        temp_keys[k+1] = node_wram->keys[k];
        temp_children[k+2] = node_wram->children[k+1];
    }
    
    int total_keys = MAX_KEYS + 1;
    int split_idx = total_keys / 2;
    *up_key = temp_keys[split_idx];
    
    node_wram->num_keys = split_idx;
    for(int k=0; k<split_idx; k++) {
        node_wram->keys[k] = temp_keys[k];
        node_wram->children[k] = temp_children[k];
    }
    node_wram->children[split_idx] = temp_children[split_idx];
    
    new_node_wram.type = INTERNAL_NODE;
    new_node_wram.num_keys = total_keys - split_idx - 1;
    for(int k=0; k<new_node_wram.num_keys; k++) {
        new_node_wram.keys[k] = temp_keys[split_idx + 1 + k];
        new_node_wram.children[k] = temp_children[split_idx + 1 + k];
    }
    new_node_wram.children[new_node_wram.num_keys] = temp_children[total_keys];
    
    *new_node_out = new_node_addr;
    
    mram_write(&new_node_wram, new_node_addr, sizeof(InternalNode));
    mram_write(node_wram, node_addr, sizeof(InternalNode));
}

static int find_child_index(InternalNode *internal, KeyType key) {
    int left = 0;
    int right = internal->num_keys - 1;
    int result = internal->num_keys;  
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (internal->keys[mid] <= key) {
            left = mid + 1;
        } else {
            result = mid;
            right = mid - 1;
        }
    }
    return result;
}

static bool insert_recursive(__mram_ptr void *node_addr, KeyType key, ValueType value, KeyType *up_key, __mram_ptr void **new_child) {
    NodeType type = get_node_type(node_addr);
    
    if (type == LEAF_NODE) {
        LeafNode leaf __dma_aligned;
        mram_read(node_addr, &leaf, sizeof(LeafNode));

        for(int i=0; i<leaf.num_keys; i++) {
            if (leaf.keys[i] == key) {
                leaf.values[i] = value;
                mram_write(&leaf, node_addr, sizeof(LeafNode));
                return false;
            }
        }
        
        if (leaf.num_keys < MAX_KEYS) {
            int pos = 0;
            while (pos < leaf.num_keys && leaf.keys[pos] < key) pos++;
            for (int k = leaf.num_keys; k > pos; k--) {
                leaf.keys[k] = leaf.keys[k-1];
                leaf.values[k] = leaf.values[k-1];
            }
            leaf.keys[pos] = key;
            leaf.values[pos] = value;
            leaf.num_keys++;
            mram_write(&leaf, node_addr, sizeof(LeafNode));
            return false;
        } else {
            __mram_ptr LeafNode *new_leaf;
            split_leaf_node((__mram_ptr LeafNode*)node_addr, &leaf, key, value, up_key, &new_leaf);
            *new_child = (__mram_ptr void*)new_leaf;
            return true;
        }
    } else {
        InternalNode internal __dma_aligned;
        mram_read(node_addr, &internal, sizeof(InternalNode));
        
        int idx = find_child_index(&internal, key);
        __mram_ptr void *child_addr = internal.children[idx];
        
        KeyType child_up_key;
        __mram_ptr void *child_new_node = NULL;
        bool split = insert_recursive(child_addr, key, value, &child_up_key, &child_new_node);
        
        if (split) {
            if (internal.num_keys < MAX_KEYS) {
                int pos = 0;
                while (pos < internal.num_keys && internal.keys[pos] < child_up_key) pos++;
                for(int k=internal.num_keys; k>pos; k--) {
                    internal.keys[k] = internal.keys[k-1];
                    internal.children[k+1] = internal.children[k];
                }
                internal.keys[pos] = child_up_key;
                internal.children[pos+1] = child_new_node;
                internal.num_keys++;
                mram_write(&internal, node_addr, sizeof(InternalNode));
                return false;
            } else {
                __mram_ptr InternalNode *new_internal;
                split_internal_node((__mram_ptr InternalNode*)node_addr, &internal, child_up_key, child_new_node, up_key, &new_internal);
                *new_child = (__mram_ptr void*)new_internal;
                return true;
            }
        }
        return false;
    }
}

bool bptree_insert(KeyType key, ValueType value) {
    if (tree_root == NULL) {
        __mram_ptr LeafNode *root = create_leaf_node();
        LeafNode root_wram __dma_aligned;
        // mram_read(root, &root_wram, sizeof(LeafNode));
        
        root_wram.keys[0] = key;
        root_wram.values[0] = value;
        root_wram.num_keys = 1;
        
        mram_write(&root_wram, root, sizeof(LeafNode));
        
        tree_root = (__mram_ptr void*)root;
        tree_height = 1;
        return true;
    }
    
    KeyType up_key;
    __mram_ptr void *new_child = NULL;
    bool split = insert_recursive(tree_root, key, value, &up_key, &new_child);
    
    if (split) {
        __mram_ptr InternalNode *new_root = create_internal_node();
        InternalNode new_root_wram __dma_aligned;
        
        new_root_wram.type = INTERNAL_NODE;
        new_root_wram.children[0] = tree_root;
        new_root_wram.keys[0] = up_key;
        new_root_wram.children[1] = new_child;
        new_root_wram.num_keys = 1;
        
        mram_write(&new_root_wram, new_root, sizeof(InternalNode));
        
        tree_root = (__mram_ptr void*)new_root;
        tree_height++;
    }
    
    return true;
}

__mram_ptr void *mram_alloc(const size_t n)
{
  __mram_ptr void* p = mram_free;
  mram_free += (n + 7) & ~7;
  return p;
}

void print_node_recursive(__mram_ptr void* node_addr, int level) {
    if (node_addr == NULL) return;
    
    NodeType type = get_node_type(node_addr);

    for(int i=0; i<level; i++) printf("  ");
    
    if (type == LEAF_NODE) {
        LeafNode node __dma_aligned;
        mram_read(node_addr, &node, sizeof(LeafNode));
        printf("[Leaf] Addr: %p, Keys: ", node_addr);
        for(int i=0; i<node.num_keys; i++) printf("%d ", node.keys[i]);
        printf("\n");
    } else {
        InternalNode node __dma_aligned;
        mram_read(node_addr, &node, sizeof(InternalNode));
        printf("[Internal] Addr: %p, Keys: ", node_addr);
        for(int i=0; i<node.num_keys; i++) printf("%d ", node.keys[i]);
        printf("\n");
        
        for(int i=0; i<=node.num_keys; i++) {
            print_node_recursive(node.children[i], level + 1);
        }
    }
}

void print_tree_structure() {
    printf("--- Tree Structure ---\n");
    print_node_recursive(tree_root, 0);
    printf("----------------------\n");
}

int main()
{
  if (me() == 0) {
    int limit = 15;
    if (limit > nr_queries) limit = nr_queries;

    printf("Initial Tree:\n");
    print_tree_structure();

    for (int i = 0; i < limit; i++) {
      printf("\n>>> Inserting Key: %d\n", query_buffer[i].key);
      bptree_insert(query_buffer[i].key, query_buffer[i].value);
      print_tree_structure();
    }
  }
  return 0;
}

#include "btree.h"
#include <mram.h>
#include <alloc.h>
#include <stdalign.h>
#include <mutex.h>
#include <string.h>
#include <assert.h>
#include <stddef.h>
#include <stdio.h>

/* Configs */
#define DEGREE (20)
#define MAX_DEPTH (16)
#define DEGREE_RIGHT ((DEGREE) / 2)
#define DEGREE_LEFT ((DEGREE) - (DEGREE_RIGHT))

/* Types */
typedef uint32_t node_id_t;
#define node_id_null ((node_id_t)-1)
typedef struct _node_t {
  btree_key_t keys[DEGREE];
  union {
    node_id_t children[DEGREE+1];
    struct {
      btree_val_t values[DEGREE];
      node_id_t next;
    } leaf;
  } arr;
  uint16_t num_keys;
  bool is_leaf;
} node_t;
static_assert(sizeof(node_t) % 8 == 0, "");

typedef struct _desc_t {
  node_id_t root;
  bool allow_duplicates;
} desc_t;

/* Node allocator */
#define BTREE_ALLOCATOR_SIZE_BITS (16)
#define BTREE_ALLOCATOR_SIZE (1UL << BTREE_ALLOCATOR_SIZE_BITS) // 64K
static __mram_noinit node_t node_alloc_pool[BTREE_ALLOCATOR_SIZE];
MUTEX_INIT(node_alloc_mutex);
static node_id_t node_alloc_next;

static inline node_id_t node_alloc() {
  mutex_lock(node_alloc_mutex);
  node_id_t node_id = node_alloc_next;
  ++node_alloc_next;
  mutex_unlock(node_alloc_mutex);
  return node_id;
}

static inline __mram_ptr node_t *node_get(node_id_t node_id) {
  return &node_alloc_pool[node_id];
}

static inline void node_read(node_t *wram_buf, node_id_t node_id) {
  mram_read(node_get(node_id), wram_buf, sizeof(node_t));
}

static inline void node_write(node_t *wram_buf, node_id_t node_id) {
  mram_write(wram_buf, node_get(node_id), sizeof(node_t));
}

/* Lock manager for synchronizing writers */
#define LOCK_HASHMAP_SIZE_BITS (7)
#define LOCK_HASHMAP_SIZE (1UL << LOCK_HASHMAP_SIZE_BITS)
#define LOCK_HASH_MASK (LOCK_HASHMAP_SIZE - 1)
static volatile node_id_t lock_manager_hashmap[LOCK_HASHMAP_SIZE];
MUTEX_INIT(lock_manager_mutex);
static void lock_init_global() {
  for (uint32_t i = 0; i < LOCK_HASHMAP_SIZE; i++) {
    lock_manager_hashmap[i] = node_id_null;
  }
}

static uint32_t node_lock(node_id_t node_id) {
  uint32_t hash_id;
  bool acquired;
  while (true) {
    mutex_lock(lock_manager_mutex);
    hash_id = node_id & LOCK_HASH_MASK;
    while (true) {
      uint32_t element = lock_manager_hashmap[hash_id];
      if (element == node_id_null) {
        acquired = true;
        break;
      }
      if (element == node_id) {
        acquired = false;
        break;
      }
      hash_id = (hash_id + 1) & LOCK_HASH_MASK; // linear probing
    }
    if (acquired) {
      lock_manager_hashmap[hash_id] = node_id;
      mutex_unlock(lock_manager_mutex);
      break;
    }
    mutex_unlock(lock_manager_mutex);
    while (lock_manager_hashmap[hash_id] == node_id);
  }
  return hash_id;
}

static void node_unlock(uint32_t handle) {
  mutex_lock(lock_manager_mutex);
  lock_manager_hashmap[handle] = node_id_null;
  mutex_unlock(lock_manager_mutex);
}

/* Helpers */
static uint16_t node_traverse(node_t *node, btree_key_t key) {
  uint16_t low = 0, high = node->num_keys;
  while (low < high) {
    uint16_t mid = (low + high) / 2;
    if (key <= node->keys[mid]) high = mid; // key <= mid
    else low = mid + 1; // mid < key
  }
  return low;
}

static_assert(sizeof(btree_val_t) == sizeof(node_id_t), "");
static void node_insert(node_t *node, uint16_t slot, btree_key_t key, node_id_t val) {
  int16_t num_keys = (int16_t)node->num_keys;
  bool is_leaf = node->is_leaf;
  btree_key_t *keys = &node->keys[0];
  node_id_t *array = is_leaf ? &node->arr.leaf.values[0] : &node->arr.children[1];
  for (int16_t i = num_keys - 1; i >= (int16_t)slot; --i) {
    keys[i + 1] = keys[i];
    array[i + 1] = array[i];
  }
  keys[slot] = key;
  array[slot] = val;
  ++node->num_keys;
}

/* Functions */
void btree_init_global() {
  node_alloc_next = 0;
  lock_init_global();
}

void btree_init(btree_t *bt, bool allow_duplicates) {
  __dma_aligned node_t node_buf;
  // Initialize root
  node_buf.num_keys = 0;
  node_buf.is_leaf = true;
  node_buf.arr.leaf.next = node_id_null;
  node_id_t root_id = node_alloc();
  node_write(&node_buf, root_id);

  // Initialize descriptor
  desc_t *bt_desc = mem_alloc(sizeof(desc_t));
  bt_desc->root = root_id;
  bt_desc->allow_duplicates = allow_duplicates;
  *bt = bt_desc;
}

btree_val_t btree_get(btree_t bt, btree_key_t key) {
  __dma_aligned node_t node_buf;
  node_id_t node_id;
  uint16_t slot;

  // Traverse
  node_id = ((desc_t*)bt)->root;
  while (true) {
    node_read(&node_buf, node_id);
    slot = node_traverse(&node_buf, key);
    if (node_buf.is_leaf) break;
    node_id = node_buf.arr.children[slot];
  }

  // Return value
  return (slot < DEGREE && key == node_buf.keys[slot]) ?
    node_buf.arr.leaf.values[slot] : BTREE_NOVAL;
}

uint32_t btree_scan(btree_t bt, bool range, btree_key_t *key,
    uint32_t max_outs, btree_val_t *out_vals) {
  __dma_aligned node_t node_buf;
  node_id_t node_id;
  uint16_t slot;
  uint32_t num_outs;

  // Traverse
  node_id = ((desc_t*)bt)->root;
  while (true) {
    node_read(&node_buf, node_id);
    slot = node_traverse(&node_buf, key[0]);
    if (node_buf.is_leaf) break;
    node_id = node_buf.arr.children[slot];
  }

  // Edge case
  if (range && !(slot < DEGREE)) {
    node_id = node_buf.arr.leaf.next;
    node_read(&node_buf, node_id);
    slot = node_traverse(&node_buf, key[0]);
  }

  // No value found
  if (!(slot < DEGREE) ||
        // range: no key in (key[0], key[1])
      (range ? (node_buf.keys[slot] > key[1])
        // !range: no key matches key[0]
      : (node_buf.keys[slot] != key[0]))
      ) {
    return 0;
  }

  // At least one value found
  num_outs = 0;
  while (true) {
    if (num_outs >= max_outs) break;
    if (range ? (node_buf.keys[slot] > key[1])
        : (node_buf.keys[slot] != key[0])
        )
      break;
    out_vals[num_outs] = node_buf.arr.leaf.values[slot];
    ++num_outs;
    ++slot;
    if (slot >= DEGREE) {
      node_id = node_buf.arr.leaf.next;
      if (node_id == node_id_null) break;
      slot = 0;
      node_read(&node_buf, node_id);
    }
  }
  return num_outs;
}

btree_val_t btree_insert(btree_t bt, btree_key_t key, btree_val_t val) {
  node_id_t node_stack[MAX_DEPTH];
  uint16_t slot_stack[MAX_DEPTH];
  uint32_t lock_stack[MAX_DEPTH];
  __dma_aligned node_t node_buf, new_node_buf;
  node_id_t node_id, new_node_id;
  uint16_t slot;
  int16_t depth;
  btree_key_t split_key = 0;
  node_id_t split_node;
  bool insert_done;
  btree_val_t ret;
  desc_t *const bt_desc = (desc_t*)bt;
  const bool allow_duplicates = bt_desc->allow_duplicates;

  // Grab root lock; bt_desc->root lock is acquired
  depth = 0;
  insert_retry_grab_rootp:
  node_id = bt_desc->root;
  lock_stack[depth] = node_lock(node_id);
  if (bt_desc->root != node_id) {
    node_unlock(lock_stack[depth]);
    goto insert_retry_grab_rootp;
  }

  // Traverse down
  while (true) {
    node_stack[depth] = node_id;
    node_read(&node_buf, node_id);
    if (node_buf.num_keys < DEGREE) {
      for (int16_t d = depth - 1; d >= 0; --d) {
        if (lock_stack[d] == (uint32_t)-1) break;
        node_unlock(lock_stack[d]);
        lock_stack[d] = (uint32_t)-1;
      }
    }
    slot = node_traverse(&node_buf, key);
    slot_stack[depth] = slot;
    if (node_buf.is_leaf) break;
    node_id = node_buf.arr.children[slot];
    ++depth;
    lock_stack[depth] = node_lock(node_id);
  }
  const int16_t max_depth = depth;
  assert(max_depth < MAX_DEPTH);

  // Insert to leaf
  if (!allow_duplicates && slot < node_buf.num_keys && node_buf.keys[slot] == key) {
    ret = node_buf.arr.leaf.values[slot]; // already exists
    goto insert_end;
  }
  split_node = node_id_null;
  insert_done = false;
  if (node_buf.num_keys == DEGREE) {
    // Leaf node is full, split
    new_node_buf.num_keys = DEGREE_RIGHT;
    new_node_buf.is_leaf = true;
    new_node_buf.arr.leaf.next = node_buf.arr.leaf.next;
    memcpy(&new_node_buf.keys[0], &node_buf.keys[DEGREE_LEFT], sizeof(btree_key_t) * DEGREE_RIGHT);
    memcpy(&new_node_buf.arr.leaf.values[0], &node_buf.arr.leaf.values[DEGREE_LEFT], sizeof(btree_val_t) * DEGREE_RIGHT);
    if (slot >= DEGREE_LEFT) {
      node_insert(&new_node_buf, slot - DEGREE_LEFT, key, val);
      insert_done = true;
    }
    new_node_id = node_alloc();
    node_write(&new_node_buf, new_node_id);
    node_buf.arr.leaf.next = new_node_id;
    node_buf.num_keys = DEGREE_LEFT;
    split_key = node_buf.keys[DEGREE_LEFT - 1];
    split_node = new_node_id;
  }
  if (!insert_done) {
    node_insert(&node_buf, slot, key, val);
  }
  node_write(&node_buf, node_id);
  --depth;

  // Split insert into inner nodes
  for (; depth >= 0; --depth) {
    if (split_node == node_id_null) break;
    // Insert (split_key, split_node) to the node
    btree_key_t next_split_key = 0;
    node_id_t next_split_node = node_id_null;
    insert_done = false;
    node_id = node_stack[depth];
    node_read(&node_buf, node_id);
    slot = slot_stack[depth];
    if (node_buf.num_keys == DEGREE) {
      // Inner node is full, split
      new_node_buf.num_keys = DEGREE_RIGHT;
      new_node_buf.is_leaf = false;
      memcpy(&new_node_buf.keys[0], &node_buf.keys[DEGREE_LEFT], sizeof(btree_key_t) * DEGREE_RIGHT);
      memcpy(&new_node_buf.arr.children[0], &node_buf.arr.children[DEGREE_LEFT], sizeof(node_id_t) * (DEGREE_RIGHT + 1));
      if (slot >= DEGREE_LEFT) {
        node_insert(&new_node_buf, slot - DEGREE_LEFT, split_key, split_node);
        insert_done = true;
      }
      new_node_id = node_alloc();
      node_write(&new_node_buf, new_node_id);
      node_buf.num_keys = DEGREE_LEFT;
      next_split_key = node_buf.keys[DEGREE_LEFT - 1];
      next_split_node = new_node_id;
    }
    if (!insert_done) {
      node_insert(&node_buf, slot, split_key, split_node);
    }
    node_write(&node_buf, node_id);
    split_key = next_split_key;
    split_node = next_split_node;
  }

  // Split root
  if (split_node != node_id_null) {
    new_node_buf.num_keys = 1;
    new_node_buf.is_leaf = false;
    new_node_buf.keys[0] = split_key;
    new_node_buf.arr.children[0] = bt_desc->root;
    new_node_buf.arr.children[1] = split_node;
    new_node_id = node_alloc();
    node_write(&new_node_buf, new_node_id);
    bt_desc->root = new_node_id;
  }

  ret = BTREE_NOVAL;
  insert_end:
  for (int16_t d = max_depth; d >= 0; --d) {
    if (lock_stack[d] == (uint32_t)-1) break;
    node_unlock(lock_stack[d]);
  }
  return ret;
}

#ifdef btree_debug
void node_print(node_t *node) {
  for (uint16_t i = 0; i < node->num_keys; node++) {
    printf("%lu ", node->keys[i]);
  }
  printf("\n");
}

void btree_print(btree_t bt) {
  __dma_aligned node_t node_buf;
  node_id_t node_id;
  uint16_t slot;

  node_id = ((desc_t*)bt)->root;
  while (true) {
    node_read(&node_buf, node_id);
    if (node_buf.is_leaf) break;
    node_id = node_buf.arr.children[0];
  }

  while (true) {
    node_print(&node_buf);
    node_id = node_buf.arr.leaf.next;
    if (node_id == node_id_null) break;
    node_read(&node_buf, node_id);
  }
}
#endif

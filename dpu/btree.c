#include "btree.h"
#include "global.h"
#include <mram.h>
#include <alloc.h>
#include <stdalign.h>
#include <mutex.h>
#include <assert.h>
#include <string.h>
#include <stddef.h>
#include <stdio.h>

/* Configs */
#define DEGREE (20)
#define MAX_DEPTH (16)
#define DEGREE_RIGHT ((DEGREE) / 2)
#define DEGREE_LEFT ((DEGREE) - (DEGREE_RIGHT))
#define BTREE_ALLOCATOR_SIZE_BITS (16)
#define BTREE_ALLOCATOR_SIZE (1UL << BTREE_ALLOCATOR_SIZE_BITS) // 64K

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
#define LOCK_HASHSET_SIZE_BITS (8)
#define LOCK_HASHSET_SIZE (1UL << LOCK_HASHSET_SIZE_BITS)
#define LOCK_HASH_MASK (LOCK_HASHSET_SIZE - 1)
static volatile node_id_t lock_manager_hashset[LOCK_HASHSET_SIZE];
MUTEX_INIT(lock_manager_mutex);
static void lock_init_global() {
  for (uint32_t i = 0; i < LOCK_HASHSET_SIZE; i++) {
    lock_manager_hashset[i] = node_id_null;
  }
}

// robin hood hashing
// if succeed, *hash_id is the id where value is inserted.
// if failed, *hash_id is the id of conflicting element
static inline bool hashset_insert_if_absent(uint32_t new_value, uint32_t *id) {
  uint32_t hash_id = new_value & LOCK_HASH_MASK;
  uint32_t insert_target = new_value;
  while (true) {
    uint32_t old_value = lock_manager_hashset[hash_id];
    if (old_value == new_value) {
      *id = hash_id;
      return false; // already in the set
    }
    if (old_value == node_id_null) {
      if (insert_target == new_value) *id = hash_id;
      lock_manager_hashset[hash_id] = insert_target;
      return true; // insert succeed
    }
    if ((((insert_target & LOCK_HASH_MASK) - hash_id) & LOCK_HASH_MASK) >
        (((old_value & LOCK_HASH_MASK) - hash_id) & LOCK_HASH_MASK)) {
      // swap
      if (insert_target == new_value) *id = hash_id;
      lock_manager_hashset[hash_id] = insert_target;
      insert_target = old_value;
    }
    hash_id = (hash_id + 1) & LOCK_HASH_MASK;
  }
}

static inline void hashset_delete(uint32_t value) {
  uint32_t hash_id = value & LOCK_HASH_MASK;
  while (lock_manager_hashset[hash_id] != value) {
    hash_id = (hash_id + 1) & LOCK_HASH_MASK;
  }
  while (true) {
    uint32_t next_hash_id = (hash_id + 1) & LOCK_HASH_MASK;
    uint32_t next_value = lock_manager_hashset[next_hash_id];
    if ((next_value == node_id_null) ||
        ((next_value & LOCK_HASH_MASK) == next_hash_id)) {
      lock_manager_hashset[hash_id] = node_id_null;
      return;
    }
    lock_manager_hashset[hash_id] = next_value;
    hash_id = next_hash_id;
  }
}

static void node_lock(node_id_t node_id) {
  while (true) {
    uint32_t id;
    mutex_lock(lock_manager_mutex);
    bool inserted = hashset_insert_if_absent(node_id, &id);
    mutex_unlock(lock_manager_mutex);
    if (inserted) break;
    while (lock_manager_hashset[id] == node_id);
  }
}

static void node_unlock(node_id_t node_id) {
  mutex_lock(lock_manager_mutex);
  hashset_delete(node_id);
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

void btree_scan(btree_t bt, btree_key_t *keys,
    btree_scan_callback_t callback, void *args) {
  const bool range_scan = (keys[0] != keys[1]);
  __dma_aligned node_t node_buf;
  node_id_t node_id;
  uint16_t slot;

  // Traverse
  node_id = ((desc_t*)bt)->root;
  while (true) {
    node_read(&node_buf, node_id);
    slot = node_traverse(&node_buf, keys[0]);
    if (node_buf.is_leaf) break;
    node_id = node_buf.arr.children[slot];
  }

  // Edge case
  if (range_scan && !(slot < DEGREE)) {
    node_id = node_buf.arr.leaf.next;
    node_read(&node_buf, node_id);
    slot = node_traverse(&node_buf, keys[0]);
  }

  // No value found
  if (!(slot < DEGREE) ||
        // range: no key in (key[0], key[1])
      (range_scan ? (node_buf.keys[slot] > keys[1])
        // !range: no key matches key[0]
      : (node_buf.keys[slot] != keys[0]))
      ) {
    return;
  }

  // At least one value found
  while (true) {
    // key end condition
    if (range_scan ? (node_buf.keys[slot] > keys[1])
        : (node_buf.keys[slot] != keys[0])
        )
      break;
    // call callback
    if (!callback(node_buf.arr.leaf.values[slot], args))
      break;
    ++slot;
    if (slot >= node_buf.num_keys) {
      node_id = node_buf.arr.leaf.next;
      if (node_id == node_id_null) break;
      slot = 0;
      node_read(&node_buf, node_id);
    }
  }
}

btree_val_t btree_insert(btree_t bt, btree_key_t key, btree_val_t val) {
  node_id_t node_stack[MAX_DEPTH];
  uint16_t slot_stack[MAX_DEPTH];
  int16_t unlocked_depth = -1;
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
  grab_rootp:
  node_id = bt_desc->root;
  node_lock(node_id);
  if (bt_desc->root != node_id) {
    node_unlock(node_id);
    goto grab_rootp;
  }

  // Traverse down
  while (true) {
    node_stack[depth] = node_id;
    node_read(&node_buf, node_id);
    if (node_buf.num_keys < DEGREE) {
      for (int16_t d = depth - 1; d > unlocked_depth; --d) {
        node_unlock(node_stack[d]);
      }
      unlocked_depth = depth - 1;
    }
    slot = node_traverse(&node_buf, key);
    slot_stack[depth] = slot;
    if (node_buf.is_leaf) break;
    node_id = node_buf.arr.children[slot];
    ++depth;
    node_lock(node_id);
  }
  const int16_t max_depth = depth;
  assert_print(max_depth < MAX_DEPTH);

  // Insert to leaf
  if (!allow_duplicates && slot < node_buf.num_keys && node_buf.keys[slot] == key) {
    ret = node_buf.arr.leaf.values[slot]; // already exists
    goto end;
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
  end:
  for (int16_t d = max_depth; d > unlocked_depth; --d) {
    node_unlock(node_stack[d]);
  }
  return ret;
}

#ifdef btree_debug
void node_print(node_t *node) {
  for (uint16_t i = 0; i < node->num_keys; node++) {
    printf("(%lu,%u) ", node->keys[i], node->arr.leaf.values[i]);
  }
  printf("\n");
}

void btree_print(btree_t bt) {
  __dma_aligned node_t node_buf;
  node_id_t node_id;
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

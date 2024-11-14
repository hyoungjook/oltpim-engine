#include "wset.h"
#include "global.h"
#include <mram.h>
#include <mutex.h>
#include <assert.h>
#include <stddef.h>

/* Configs */
#define LIST_DEGREE (30)
#define LIST_ALLOCATOR_SIZE_BITS (16)
#define LIST_ALLOCATOR_SIZE (1UL << LIST_ALLOCATOR_SIZE_BITS)
#define HASHMAP_SIZE_BITS (15)
#define HASHMAP_SIZE (1UL << HASHMAP_SIZE_BITS)

/* Write set list */
typedef uint32_t ws_node_id_t;
#define ws_node_id_null ((ws_node_id_t)-1)
typedef struct _ws_list_node_meta {
  uint32_t num_items;
  ws_node_id_t next;
} ws_list_node_meta;
static_assert(sizeof(ws_list_node_meta) == 8, "");
typedef union _ws_list_node_t {
  struct {
    oid_t items[LIST_DEGREE];
    ws_list_node_meta meta;
  } n;
  ws_node_id_t freeptr;
} ws_list_node_t;
static_assert(sizeof(ws_list_node_t) % 8 == 0, "");
static_assert((sizeof(oid_t) * LIST_DEGREE) % 8 == 0, "");

// write_set_t is the index into the allocator array
static __mram_noinit ws_list_node_t ws_alloc_pool[LIST_ALLOCATOR_SIZE];
static ws_node_id_t ws_free_list_head;
// mutex protects allocation
// wram_buf is used only under mutex
MUTEX_INIT(ws_alloc_mutex);
static __dma_aligned ws_node_id_t ws_wram_buf[2];

static inline ws_node_id_t ws_pool_read_freeptr(ws_node_id_t id) {
  mram_read(&ws_alloc_pool[id].freeptr, &ws_wram_buf, 8);
  return ws_wram_buf[0];
}

static inline void ws_pool_write_freeptr(ws_node_id_t id, ws_node_id_t ptr) {
  ws_wram_buf[0] = ptr;
  mram_write(&ws_wram_buf, &ws_alloc_pool[id].freeptr, 8);
}

static void ws_pool_init_global() {
  // Init free list
  for (ws_node_id_t idx = 0; idx < LIST_ALLOCATOR_SIZE; ++idx) {
    // each entry points to the next free slot
    ws_pool_write_freeptr(idx, idx + 1);
  }
  ws_free_list_head = 0;
}

static ws_node_id_t ws_node_alloc() {
  mutex_lock(ws_alloc_mutex);
  assert_print(ws_free_list_head != ws_node_id_null); // OOM
  ws_node_id_t new_free_list_head = ws_pool_read_freeptr(ws_free_list_head);
  ws_node_id_t new_id = ws_free_list_head;
  ws_free_list_head = new_free_list_head;
  mutex_unlock(ws_alloc_mutex);
  return new_id;
}

static void ws_node_free(ws_node_id_t id) {
  mutex_lock(ws_alloc_mutex);
  ws_node_id_t free_list_second = ws_pool_read_freeptr(ws_free_list_head);
  ws_pool_write_freeptr(id, free_list_second);
  ws_free_list_head = id;
  mutex_unlock(ws_alloc_mutex);
}

static inline void ws_node_read_meta(ws_node_id_t id, ws_list_node_meta *meta) {
  mram_read(&ws_alloc_pool[id].n.meta, meta, sizeof(ws_list_node_meta));
}

static inline void ws_node_write_meta(ws_node_id_t id, ws_list_node_meta *meta) {
  mram_write(meta, &ws_alloc_pool[id].n.meta, sizeof(ws_list_node_meta));
}

static inline void ws_node_write_entry(ws_node_id_t id, uint32_t idx, oid_t entry) {
  __dma_aligned oid_t buf[2];
  mram_read(&ws_alloc_pool[id].n.items[idx & (~0x1)], &buf, 8);
  buf[idx & 0x1] = entry;
  mram_write(&buf, &ws_alloc_pool[id].n.items[idx & (~0x1)], 8);
}

// If list is null, create new node and add oid
// if not null, add to the list
static ws_node_id_t ws_list_add(ws_node_id_t list, oid_t oid) {
  __dma_aligned ws_list_node_meta meta_buf;
  if (list != ws_node_id_null) {
    ws_node_read_meta(list, &meta_buf);
    uint32_t num_items = meta_buf.num_items;
    if (num_items < LIST_DEGREE) {
      ++meta_buf.num_items;
      ws_node_write_meta(list, &meta_buf);
      ws_node_write_entry(list, num_items, oid);
      return list;
    }
  }
  ws_node_id_t new_head = ws_node_alloc();
  ws_node_write_entry(new_head, 0, oid);
  meta_buf.num_items = 1;
  meta_buf.next = list;
  ws_node_write_meta(new_head, &meta_buf);
  return new_head;
}

/* Hash map */
#define HASHMAP_MASK (HASHMAP_SIZE - 1)
#define key_null ((xid_t)-1)
typedef struct _xid_map_pair_t {
  xid_t key;
  ws_node_id_t value;
} xid_map_pair_t;
static_assert(sizeof(xid_map_pair_t) % 8 == 0, "");
static __mram_noinit xid_map_pair_t xid_map[HASHMAP_SIZE];
MUTEX_INIT(xid_map_mutex);
static __dma_aligned xid_map_pair_t xid_pair_buf[2];

static void xid_map_init_global() {
  for (uint32_t i = 0; i < HASHMAP_SIZE; i++) {
    xid_map[i].key = key_null;
  }
}

static inline void xid_pair_read(uint32_t hash_id, xid_map_pair_t *buf) {
  mram_read(&xid_map[hash_id], buf, sizeof(xid_map_pair_t));
}

static inline void xid_pair_write(uint32_t hash_id, xid_map_pair_t *buf) {
  mram_write(buf, &xid_map[hash_id], sizeof(xid_map_pair_t));
}

// (old_value, arg) -> new_value
typedef ws_node_id_t (*insert_callback)(ws_node_id_t, uint32_t);

// robin hood hashing
// if not found, call callback(ws_node_id_null) and store its return value
// if found, call callback(old_value) and update to its return value
static inline bool xid_map_insert(xid_t key, insert_callback callback, uint32_t arg) {
  uint32_t hash_id = key & HASHMAP_MASK;
  xid_map_pair_t *const wram_buf = &xid_pair_buf[0];
  xid_map_pair_t *const insert_target = &xid_pair_buf[1];
  insert_target->key = key;
  while (true) {
    xid_pair_read(hash_id, wram_buf);
    if (wram_buf->key == key) {
      ws_node_id_t new_val = callback(wram_buf->value, arg);
      if (new_val != wram_buf->value) {
        wram_buf->value = new_val;
        xid_pair_write(hash_id, wram_buf);
      }
      return false; // already in the map
    }
    if (wram_buf->key == key_null) {
      if (insert_target->key == key) {
        insert_target->value = callback(ws_node_id_null, arg);
      }
      xid_pair_write(hash_id, insert_target);
      return true; // insert succeed
    }
    if ((((insert_target->key & HASHMAP_MASK) - hash_id) & HASHMAP_MASK) >
        (((wram_buf->key & HASHMAP_MASK) - hash_id) & HASHMAP_MASK)) {
      // swap
      if (insert_target->key == key) {
        insert_target->value = callback(ws_node_id_null, arg);
      }
      xid_pair_write(hash_id, insert_target);
      insert_target->key = wram_buf->key;
      insert_target->value = wram_buf->value;
    }
    hash_id = (hash_id + 1) & HASHMAP_MASK;
  }
}

static inline ws_node_id_t xid_map_get_delete(xid_t key) {
  uint32_t hash_id = key & HASHMAP_MASK;
  xid_map_pair_t *const wram_buf = &xid_pair_buf[0];
  while (true) {
    xid_pair_read(hash_id, wram_buf);
    if (wram_buf->key == key_null) {
      return ws_node_id_null; // not found
    }
    if (wram_buf->key == key) break;
    hash_id = (hash_id + 1) & HASHMAP_MASK;
  }
  const ws_node_id_t value = wram_buf->value;
  while (true) {
    uint32_t next_hash_id = (hash_id + 1) & HASHMAP_MASK;
    xid_pair_read(next_hash_id, wram_buf); // next_value
    if ((wram_buf->key == key_null) ||
        ((wram_buf->key & HASHMAP_MASK) == next_hash_id)) {
      wram_buf->key = key_null;
      xid_pair_write(hash_id, wram_buf);
      return value;
    }
    xid_pair_write(hash_id, wram_buf);
    hash_id = next_hash_id;
  }
}

/* Functions */
void wset_init_global() {
  ws_pool_init_global();
  xid_map_init_global();
}

void wset_add(xid_t xid, oid_t oid) {
  mutex_lock(xid_map_mutex);
  xid_map_insert(xid, ws_list_add, oid);
  mutex_unlock(xid_map_mutex);
}

void wset_traverse_remove(xid_t xid, wset_callback callback, void *arg) {
  mutex_lock(xid_map_mutex);
  ws_node_id_t list = xid_map_get_delete(xid);
  mutex_unlock(xid_map_mutex);

  __dma_aligned ws_list_node_t node_buf;
  while (list != ws_node_id_null) {
    mram_read(&ws_alloc_pool[list], &node_buf, sizeof(ws_list_node_t));
    ws_node_free(list);
    for (uint32_t i = 0; i < node_buf.n.meta.num_items; ++i) {
      callback(node_buf.n.items[i], arg);
    }
    if (node_buf.n.meta.next == ws_node_id_null) break;
    list = node_buf.n.meta.next;
  }
}

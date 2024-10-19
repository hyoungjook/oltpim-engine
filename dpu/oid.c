#include "oid.h"
#include <mram.h>
#include <mutex.h>
#include <assert.h>

/* Configs */
#define OID_ARRAY_SIZE_BITS (17)
#define OID_ARRAY_SIZE (1UL << OID_ARRAY_SIZE_BITS)
#define MAX_OID (OID_ARRAY_SIZE - 1)

/* Types */
#define oid_index_null ((oid_t)-1)

/* OID Array */
static_assert(sizeof(oid_value_t) == 4, "");
static __mram_noinit oid_value_t oid_array[OID_ARRAY_SIZE];
oid_t oid_free_list_head;
MUTEX_INIT(oid_array_mutex);
static __dma_aligned oid_value_t oid_wram_buf[2];

void oid_manager_init_global() {
  // Init free list
  __dma_aligned oid_value_t wram_buf[8];
  for (oid_t idx = 0; idx < OID_ARRAY_SIZE; idx += 8) {
    for (oid_t i = 0; i < 8; ++i) {
      oid_value_t next_free_oid = idx + i + 1;
      wram_buf[i] = next_free_oid;
    }
    if (idx == OID_ARRAY_SIZE - 8) wram_buf[7] = oid_index_null;
    mram_write(wram_buf, &oid_array[idx], sizeof(oid_value_t) * 8);
  }
  oid_free_list_head = 0;
}

static inline oid_value_t oid_array_read(oid_t oid) {
  // 8 / sizeof(oid_value_t) == 2, so align with 2 entries
  mram_read(&oid_array[oid & (~0x1)], oid_wram_buf, 8);
  return oid_wram_buf[oid & 0x1];
}

static inline void oid_array_write_after_read(oid_t oid, oid_value_t val) {
  // oid_wram_buf already contains the value
  oid_wram_buf[oid & 0x1] = val;
  mram_write(oid_wram_buf, &oid_array[oid & (~0x1)], 8);
}

oid_t oid_alloc() {
  mutex_lock(oid_array_mutex);
  if (oid_free_list_head == oid_index_null) {
    assert(false); // OOM
  }
  oid_value_t new_free_list_head = oid_array_read(oid_free_list_head);
  oid_t new_oid = oid_free_list_head;
  oid_free_list_head = new_free_list_head;
  mutex_unlock(oid_array_mutex);
  return new_oid;
}

void oid_free(oid_t oid) {
  mutex_lock(oid_array_mutex);
  oid_value_t free_list_second = oid_array_read(oid_free_list_head);
  oid_array_read(oid);
  oid_array_write_after_read(oid, free_list_second);
  oid_free_list_head = oid;
  mutex_unlock(oid_array_mutex);
}

void oid_set(oid_t oid, oid_value_t val) {
  mutex_lock(oid_array_mutex);
  oid_array_read(oid);
  oid_array_write_after_read(oid, val);
  mutex_unlock(oid_array_mutex);
}

oid_value_t oid_get(oid_t oid) {
  mutex_lock(oid_array_mutex);
  oid_value_t val = oid_array_read(oid);
  mutex_unlock(oid_array_mutex);
  return val;
}

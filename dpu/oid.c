#include "oid.h"
#include "global.h"
#include <mram.h>
#include <mutex.h>
#include <assert.h>
#include <stddef.h>

/* Configs */
#define OID_ARRAY_SIZE (1UL << OID_ARRAY_SIZE_BITS)
#define MAX_OID (OID_ARRAY_SIZE - 1)

/* OID Array */
static_assert(sizeof(oid_value_t) == 4, "");
static __mram_noinit oid_value_t oid_array[OID_ARRAY_SIZE];
static oid_t oid_free_list_head;
// oid_buf is used only under oid_array_mutex.
MUTEX_INIT(oid_array_mutex);
static __dma_aligned oid_value_t oid_buf[2];

/* Helpers */
static oid_value_t oid_array_read(oid_t oid) {
  mram_read(&oid_array[oid & (~0x1)], &oid_buf, 8);
  return oid_buf[oid & 0x1];
}

static void oid_array_write_after_read(oid_t oid, oid_value_t val) {
  // assume oid_array_read(oid) is called and oid_buf is filled
  oid_buf[oid & 0x1] = val;
  mram_write(&oid_buf, &oid_array[oid & (~0x1)], 8);
}

static void oid_array_write(oid_t oid, oid_value_t val) {
  oid_array_read(oid);
  oid_array_write_after_read(oid, val);
}

/* Functions */
void oid_manager_init_global() {
  // Init free list
  __dma_aligned oid_value_t buf[2];
  for (oid_t idx = 0; idx < OID_ARRAY_SIZE; idx += 2) {
    // each entry points to the next free oid
    buf[0] = idx + 1;
    buf[1] = idx + 2;
    mram_write(&buf, &oid_array[idx], 8);
  }
  buf[1] = oid_value_null;
  mram_write(&buf, &oid_array[OID_ARRAY_SIZE - 1], 8);
  oid_free_list_head = 0;
}

oid_t oid_alloc_set(oid_value_t val) {
  mutex_lock(oid_array_mutex);
  // pop free list and alloc one
  always_assert(oid_free_list_head != oid_value_null); // OOM
  oid_value_t new_free_list_head = oid_array_read(oid_free_list_head);
  oid_t new_oid = oid_free_list_head;
  oid_free_list_head = new_free_list_head;
  // initialize value
  oid_array_write(new_oid, val);
  mutex_unlock(oid_array_mutex);
  return new_oid;
}

void oid_free(oid_t oid) {
  mutex_lock(oid_array_mutex);
  // push to free list
  oid_array_write(oid, oid_free_list_head);
  oid_free_list_head = oid;
  mutex_unlock(oid_array_mutex);
}

oid_value_t oid_get(oid_t oid) {
  mutex_lock(oid_array_mutex);
  oid_value_t val = oid_array_read(oid);
  mutex_unlock(oid_array_mutex);
  return val;
}

void oid_set(oid_t oid, oid_value_t val) {
  mutex_lock(oid_array_mutex);
  oid_array_write(oid, val);
  mutex_unlock(oid_array_mutex);
}

bool oid_compare_exchange(oid_t oid, oid_value_t *old_val, oid_value_t new_val) {
  mutex_lock(oid_array_mutex);
  oid_value_t val = oid_array_read(oid);
  bool succeed = (val == *old_val);
  if (succeed) {
    oid_array_write_after_read(oid, new_val);
  }
  else {
    *old_val = val;
  }
  mutex_unlock(oid_array_mutex);
  return succeed;
}

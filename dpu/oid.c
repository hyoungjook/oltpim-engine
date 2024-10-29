#include "oid.h"
#include <mram.h>
#include <mutex.h>
#include <assert.h>
#include <stddef.h>

/* Configs */
#define OID_ARRAY_SIZE_BITS (16)
#define OID_ARRAY_SIZE (1UL << OID_ARRAY_SIZE_BITS)
#define MAX_OID (OID_ARRAY_SIZE - 1)

/* OID Array */
#define xid_null ((xid_t)-1)
typedef union _oid_entry_t {
  struct _oid_entry_real {
    xid_t xid;
    oid_value_t value;
  } data;
  oid_value_t freeptr;
} oid_entry_t;
static_assert(sizeof(oid_entry_t) % 8 == 0, "");

static __mram_noinit oid_entry_t oid_array[OID_ARRAY_SIZE];
static oid_t oid_free_list_head;
// All accesses to oid_array is protected by this mutex,
// so it is safe to use a single global oid_wram_buf
MUTEX_INIT(oid_array_mutex);
static __dma_aligned oid_entry_t oid_wram_buf;

/* Helpers */
static inline oid_value_t oid_array_read_freeptr(oid_t oid) {
  mram_read(&oid_array[oid].freeptr, &oid_wram_buf.freeptr, 8);
  return oid_wram_buf.freeptr;
}

static inline void oid_array_write_freeptr(oid_t oid, oid_value_t freeptr) {
  oid_wram_buf.freeptr = freeptr;
  mram_write(&oid_wram_buf.freeptr, &oid_array[oid].freeptr, 8);
}

static inline xid_t oid_array_read_xid(oid_t oid) {
  mram_read(&oid_array[oid].data.xid, &oid_wram_buf.data.xid, sizeof(xid_t));
  return oid_wram_buf.data.xid;
}

static inline void oid_array_write_xid(oid_t oid, xid_t xid) {
  oid_wram_buf.data.xid = xid;
  mram_write(&oid_wram_buf.data.xid, &oid_array[oid].data.xid, sizeof(xid_t));
}

static inline oid_value_t oid_array_read_value(oid_t oid) {
  mram_read(&oid_array[oid].data.value, &oid_wram_buf.data.value, 8);
  return oid_wram_buf.data.value;
}

static inline void oid_array_write_value(oid_t oid, oid_value_t value) {
  oid_wram_buf.data.value = value;
  mram_write(&oid_wram_buf.data.value, &oid_array[oid].data.value, 8);
}

/* Functions */
void oid_manager_init_global() {
  // Init free list
  for (oid_t idx = 0; idx < OID_ARRAY_SIZE; idx += 8) {
    // each entry points to the next free oid
    oid_array_write_freeptr(idx, idx + 1);
  }
  oid_free_list_head = 0;
}

oid_t oid_alloc_acquire(xid_t xid) {
  mutex_lock(oid_array_mutex);
  if (oid_free_list_head == oid_value_null) {
    assert(false); // OOM
  }
  oid_value_t new_free_list_head = oid_array_read_freeptr(oid_free_list_head);
  oid_t new_oid = oid_free_list_head;
  oid_free_list_head = new_free_list_head;
  { // write xid and value at once
    oid_wram_buf.data.xid = xid;
    oid_wram_buf.data.value = oid_value_null;
    mram_write(&oid_wram_buf, &oid_array[new_oid], sizeof(oid_entry_t));
  }
  mutex_unlock(oid_array_mutex);
  return new_oid;
}

void oid_free_release(oid_t oid) {
  mutex_lock(oid_array_mutex);
  oid_value_t free_list_second = oid_array_read_freeptr(oid_free_list_head);
  oid_array_write_freeptr(oid, free_list_second);
  oid_free_list_head = oid;
  mutex_unlock(oid_array_mutex);
}

bool oid_try_acquire(oid_t oid, xid_t xid) {
  bool succeed = true;
  mutex_lock(oid_array_mutex);
  xid_t prev_xid = oid_array_read_xid(oid);
  if (prev_xid != xid_null) {
    // If prev_xid == xid, succeed without write
    // Else, fail
    succeed = (prev_xid == xid);
    goto unlock;
  }
  // prev_xid is null, so write my xid
  oid_array_write_xid(oid, xid);
  unlock:
  mutex_unlock(oid_array_mutex);
  return succeed;
}

void oid_release(oid_t oid) {
  mutex_lock(oid_array_mutex);
  oid_array_write_xid(oid, xid_null);
  mutex_unlock(oid_array_mutex);
}

void oid_set(oid_t oid, oid_value_t val) {
  mutex_lock(oid_array_mutex);
  oid_array_write_value(oid, val);
  mutex_unlock(oid_array_mutex);
}

oid_value_t oid_get(oid_t oid) {
  mutex_lock(oid_array_mutex);
  oid_value_t val = oid_array_read_value(oid);
  mutex_unlock(oid_array_mutex);
  return val;
}

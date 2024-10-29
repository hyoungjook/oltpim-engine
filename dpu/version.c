#include "version.h"
#include <mram.h>
#include <assert.h>
#include <mutex.h>
#include <mutex_pool.h>

/* Configs */
#define VERSION_ALLOCATOR_SIZE_BITS (18)
#define VERSION_ALLOCATOR_SIZE (1UL << VERSION_ALLOCATOR_SIZE_BITS)
#define VERSION_ACCESS_MUTEX_NUM (16)

/* Types */
typedef union _object_t {
  struct _object_real {
    csn_t csn;
    version_value_t value;
    version_t next;
    bool dirty;   // is this a dirty version from an uncommitted txn?
  } obj;
  version_t freeptr;
} object_t;
static_assert(sizeof(object_t) % 8 == 0, "");

/* Allocator */
// version_t is the index into the allocator array
static __mram_noinit object_t ver_alloc_pool[VERSION_ALLOCATOR_SIZE];
static version_t ver_free_list_head;
// ver_alloc_mutex protects allocation
// ver_wram_buf is used only under ver_alloc_mutex
MUTEX_INIT(ver_alloc_mutex);
static __dma_aligned version_t ver_wram_buf[2];

/* Atomic access mutex */
MUTEX_POOL_INIT(ver_access_mutexes, VERSION_ACCESS_MUTEX_NUM);

/* Helpers */
static inline version_t ver_pool_read_freeptr(version_t ver) {
  mram_read(&ver_alloc_pool[ver].freeptr, &ver_wram_buf, 8);
  return ver_wram_buf[0];
}

static inline void ver_pool_write_freeptr(version_t ver, version_t freeptr) {
  ver_wram_buf[0] = freeptr;
  mram_write(&ver_wram_buf, &ver_alloc_pool[ver].freeptr, 8);
}

static version_t version_obj_alloc() {
  mutex_lock(ver_alloc_mutex);
  if (ver_free_list_head == version_null) {
    assert(false); // OOM
  }
  version_t new_free_list_head = ver_pool_read_freeptr(ver_free_list_head);
  version_t new_ver = ver_free_list_head;
  ver_free_list_head = new_free_list_head;
  mutex_unlock(ver_alloc_mutex);
  return new_ver;
}

static void version_obj_free(version_t ver) {
  // Acquire access lock of ver to prevent reading the object during freeing
  mutex_pool_lock(&ver_access_mutexes, ver);
  mutex_lock(ver_alloc_mutex);
  version_t free_list_second = ver_pool_read_freeptr(ver_free_list_head);
  ver_pool_write_freeptr(ver, free_list_second);
  ver_free_list_head = ver;
  mutex_unlock(ver_alloc_mutex);
  mutex_pool_unlock(&ver_access_mutexes, ver);
}

static inline void object_read(version_t ver, object_t *buf) {
  mutex_pool_lock(&ver_access_mutexes, ver);
  mram_read(&ver_alloc_pool[ver], buf, sizeof(object_t));
  mutex_pool_unlock(&ver_access_mutexes, ver);
}

static inline void object_write(version_t ver, object_t *buf) {
  mutex_pool_lock(&ver_access_mutexes, ver);
  mram_write(buf, &ver_alloc_pool[ver], sizeof(object_t));
  mutex_pool_unlock(&ver_access_mutexes, ver);
}

/* Functions */
void version_init_global() {
  // Init free list
  for (version_t idx = 0; idx < VERSION_ALLOCATOR_SIZE; ++idx) {
    // each entry points to the next free slot
    ver_pool_write_freeptr(idx, idx + 1);
  }
  ver_free_list_head = 0;
}

version_t version_create(csn_t csn, version_value_t new_value) {
  __dma_aligned object_t obj_buf;
  version_t new_ver = version_obj_alloc();
  obj_buf.obj.csn = csn;
  obj_buf.obj.value = new_value;
  obj_buf.obj.next = version_null;
  obj_buf.obj.dirty = true;
  // Direct write without mutex, as I am the only tasklet accessing this
  //object_write(new_ver, &obj_buf);
  mram_write(&obj_buf, &ver_alloc_pool[new_ver], sizeof(object_t));
  return new_ver;
}

bool version_read(version_t ver, csn_t csn, version_value_t *value) {
  __dma_aligned object_t obj_buf;
  version_t v = ver;
  // Traverse the version chain
  while (v != version_null) {
    object_read(v, &obj_buf);
    if ((!obj_buf.obj.dirty) && obj_buf.obj.csn <= csn) {
      // Visible version
      if (value) *value = obj_buf.obj.value;
      return true;
    }
    // Invisible, try next
    v = obj_buf.obj.next;
  }
  // No visible version found
  return false;
}

version_t version_update(version_t ver, csn_t csn, version_value_t new_value) {
  __dma_aligned object_t obj_buf;
  assert(ver != version_null);
  // Get the old head data
  object_read(ver, &obj_buf);
  if (obj_buf.obj.dirty) {
    // It's being modified by me: in-place update
    assert(obj_buf.obj.csn == csn);
    obj_buf.obj.value = new_value;
    object_write(ver, &obj_buf);
    return ver;
  }
  else {
    // First time updating this chain
    assert(obj_buf.obj.csn <= csn);
    version_t new_ver = version_obj_alloc();
    obj_buf.obj.csn = csn;
    obj_buf.obj.value = new_value;
    obj_buf.obj.next = ver;
    obj_buf.obj.dirty = true;
    object_write(new_ver, &obj_buf);
    return new_ver;
  }
}

version_t version_finalize(version_t ver, csn_t csn, bool commit) {
  __dma_aligned object_t obj_buf;
  assert(ver != version_null);
  // Get the head data
  object_read(ver, &obj_buf);
  assert(obj_buf.obj.dirty);
  if (commit) {
    // Expose the dirty head
    obj_buf.obj.csn = csn;
    obj_buf.obj.dirty = false;
    object_write(ver, &obj_buf);
    return ver;
  }
  else {
    // Remove the dirty head
    version_t new_head = obj_buf.obj.next;
    version_obj_free(ver);
    return new_head;
  }
}

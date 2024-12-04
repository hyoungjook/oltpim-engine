#include "object.h"
#include "global.h"
#include <mram.h>
#include <assert.h>
#include <mutex.h>
#include <mutex_pool.h>
#include <stddef.h>
#include "gc.h"

/* Configs */
#define VERSION_ALLOCATOR_SIZE (1UL << VERSION_ALLOCATOR_SIZE_BITS)
#define VERSION_ACCESS_MUTEX_NUM (16)

/* Types */
typedef uint32_t version_id_t;
#define version_id_null ((version_id_t)-1)
static_assert(sizeof(version_id_t) == sizeof(oid_value_t), "");

typedef struct _version_meta {
  // this is used to distinguish between free and allocated slot
  uint8_t is_free_slot;   // Whether the slot is free
  // below is used for real version obj
  uint8_t dirty;        // is this a dirty version from an uncommitted txn?
  uint8_t deleted;      // is this a deleted object?
  uint8_t pad;
} version_meta;
static_assert(sizeof(version_meta) == 4, "");

typedef struct _version_freeptr {
  version_meta meta;
  version_id_t p;
} version_freeptr;
static_assert(sizeof(version_freeptr) == 8, "");

typedef union _version_t {
  struct _version_data {
    version_meta meta;
    csn_t csn;    // xid if dirty, csn otherwise
    object_value_t value;
    version_id_t next;
  } v;
  struct _secondary_entry {
    version_meta meta;
    csn_t begin_csn, end_csn;
    object_value_t value;
  } s;
  version_freeptr freeptr;
} version_t;
static_assert(sizeof(version_t) % 8 == 0, "");
static_assert(sizeof(struct _version_freeptr) == 8, "");
static_assert(sizeof(version_t) == __VERSION_ALLOCATOR_SIZEOF_ENTITY, "");

#define SECONDARY_OID_TO_VID(oid) ((oid) & 0x7FFFFFFFUL)
#define SECONDARY_VID_TO_OID(vid) ((vid) | 0x80000000UL)
#define IS_SECONDARY_OID(oid) ((bool)((oid) & 0x80000000UL))

/* Allocator */
// version_id_t is the index into the allocator array
static __mram_noinit version_t ver_alloc_pool[VERSION_ALLOCATOR_SIZE];
static version_id_t ver_free_list_head, ver_free_list_tail;
// ver_alloc_mutex protects allocation
// ver_wram_buf is used only under ver_alloc_mutex
MUTEX_INIT(ver_alloc_mutex);
static __dma_aligned version_freeptr ver_wram_buf;

/* Atomic access mutex */
MUTEX_POOL_INIT(ver_access_mutexes, VERSION_ACCESS_MUTEX_NUM);

/* Helpers */
static inline version_id_t ver_pool_read_freeptr(version_id_t vid) {
  mram_read(&ver_alloc_pool[vid].freeptr, &ver_wram_buf, 8);
  return ver_wram_buf.p;
}

static inline void ver_pool_write_freeptr(version_id_t vid, version_id_t ptr) {
  ver_wram_buf.meta.is_free_slot = 1;
  ver_wram_buf.p = ptr;
  mram_write(&ver_wram_buf, &ver_alloc_pool[vid].freeptr, 8);
}

// Should manually mark freeptr.meta.is_free=0 after this
static version_id_t version_alloc() {
  mutex_lock(ver_alloc_mutex);
  assert_print(ver_free_list_head != ver_free_list_tail); // OOM
  version_id_t new_free_list_head = ver_pool_read_freeptr(ver_free_list_head);
  version_id_t new_vid = ver_free_list_head;
  ver_free_list_head = new_free_list_head;
  mutex_unlock(ver_alloc_mutex);
  return new_vid;
}

static void version_free(version_id_t vid) {
  // Acquire access lock of ver to prevent reading the object during freeing
  mutex_pool_lock(&ver_access_mutexes, vid);
  mutex_lock(ver_alloc_mutex);
  // Append vid to the tail of the free list
  // to spare some time before vid is reused;
  // in order to prevent subtle case between
  // object_finalize()'s abort path and object_read()
  ver_pool_write_freeptr(ver_free_list_tail, vid);
  ver_pool_write_freeptr(vid, version_id_null);
  ver_free_list_tail = vid;
  mutex_unlock(ver_alloc_mutex);
  mutex_pool_unlock(&ver_access_mutexes, vid);
}

static inline void version_read(version_id_t vid, version_t *buf) {
  mutex_pool_lock(&ver_access_mutexes, vid);
  mram_read(&ver_alloc_pool[vid], buf, sizeof(version_t));
  mutex_pool_unlock(&ver_access_mutexes, vid);
}

static inline void version_write(version_id_t vid, version_t *buf) {
  mutex_pool_lock(&ver_access_mutexes, vid);
  mram_write(buf, &ver_alloc_pool[vid], sizeof(version_t));
  mutex_pool_unlock(&ver_access_mutexes, vid);
}

/* Functions */
void object_init_global() {
  oid_manager_init_global();
  // Init free list
  for (version_id_t idx = 0; idx < VERSION_ALLOCATOR_SIZE; ++idx) {
    // each entry points to the next free slot
    ver_pool_write_freeptr(idx, idx + 1);
  }
  ver_pool_write_freeptr(VERSION_ALLOCATOR_SIZE - 1, version_id_null);
  ver_free_list_head = 0;
  ver_free_list_tail = VERSION_ALLOCATOR_SIZE - 1;
}

oid_t object_create_acquire(xid_t xid, object_value_t new_value, bool primary) {
  __dma_aligned version_t ver_buf;
  if (primary) {
    ver_buf.v.csn = xid;
    ver_buf.v.value = new_value;
    ver_buf.v.next = version_id_null;
  }
  else {
    ver_buf.s.begin_csn = xid;
    ver_buf.s.end_csn = (csn_t)-1;
    ver_buf.s.value = new_value;
  }
  ver_buf.v.meta.is_free_slot = 0;
  ver_buf.v.meta.dirty = true;
  ver_buf.v.meta.deleted = false;
  version_id_t vid = version_alloc();
  version_write(vid, &ver_buf);
  oid_t oid;
  if (primary) {
    oid = oid_alloc_set(vid);
  }
  else {
    // For secondary index, oid is just vid with MSB set to 1.
    oid = SECONDARY_VID_TO_OID(vid);
  }
  return oid;
}

void object_cancel_create(oid_t oid) {
  assert_print(!IS_SECONDARY_OID(oid));
  version_id_t vid = oid_get(oid);
  version_free(vid);
  oid_free(oid);
}

bool object_read(oid_t oid, xid_t xid, csn_t csn, object_value_t *value) {
  __dma_aligned version_t ver_buf;
  // get the vid
  version_id_t vid;
  if (!IS_SECONDARY_OID(oid)) { // primary
    retry:
    vid = oid_get(oid);
    // traverse the version chain
    while (vid != version_id_null) {
      version_read(vid, &ver_buf);
      if (ver_buf.v.meta.is_free_slot) {
        // this version is freed, retry from start
        goto retry;
      }
      if (
        (ver_buf.v.meta.dirty && (ver_buf.v.csn == xid)) || // dirty version by me
        (!ver_buf.v.meta.dirty && (ver_buf.v.csn <= csn)) // visible version
      ) {
        if (ver_buf.v.meta.deleted) {
          return false; // deleted object
        }
        if (value) *value = ver_buf.v.value;
        return true;
      }
      vid = ver_buf.v.next;
    }
    return false;
  }
  else { // secondary
    vid = SECONDARY_OID_TO_VID(oid);
    version_read(vid, &ver_buf);
    if (
        (ver_buf.s.meta.dirty) ?
          (ver_buf.s.begin_csn == xid) :  // dirty version by me
          (ver_buf.s.begin_csn <= csn && csn < ver_buf.s.end_csn) // visible version
      ) {
      if (value) *value = ver_buf.s.value;
      return true;
    }
    return false;
  }
}

status_t object_update(oid_t oid, xid_t xid, csn_t csn, object_value_t new_value,
    object_value_t *old_value, bool remove, bool *add_to_write_set) {
  assert_print(!IS_SECONDARY_OID(oid));
  __dma_aligned version_t ver_buf;
  // get the vid
  version_id_t vid;
  retry_from_oid:
  vid = oid_get(oid);
  retry_with_vid:
  if (vid == version_id_null) {
    // tried to update an invisible object
    return STATUS_FAILED;
  }
  version_read(vid, &ver_buf);
  if (ver_buf.v.meta.is_free_slot) {
    goto retry_from_oid;
  }
  if (ver_buf.v.meta.dirty) {
    if (ver_buf.v.csn == xid) {
      if (ver_buf.v.meta.deleted) {
        // I deleted this; update fails
        return STATUS_FAILED;
      }
      // My dirty version: in-place update
      if (remove) {
        ver_buf.v.meta.deleted = true;
      }
      else {
        if (old_value) *old_value = ver_buf.v.value;
        ver_buf.v.value = new_value;
      }
      // it's ensured that nobody frees this version concurrently
      // because this object is acquired by me
      version_write(vid, &ver_buf);
      *add_to_write_set = false;
      return STATUS_SUCCESS;
    }
    else {
      // Someone else's version: conflict
      return STATUS_CONFLICT;
    }
  }
  else { // clean
    if (ver_buf.v.csn <= csn && !ver_buf.v.meta.deleted) {
      // Latest version is visible: update
      if (!remove && old_value) *old_value = ver_buf.v.value;
      ver_buf.v.csn = xid;
      ver_buf.v.value = new_value;
      ver_buf.v.next = vid;
      ver_buf.v.meta.is_free_slot = 0;
      ver_buf.v.meta.dirty = true;
      ver_buf.v.meta.deleted = remove;
      version_id_t new_vid = version_alloc();
      version_write(new_vid, &ver_buf);
      // Try to install new vid
      if (!oid_compare_exchange(oid, &vid, new_vid)) {
        // conflict: retry
        version_free(new_vid);
        goto retry_with_vid;
      }
      *add_to_write_set = true;
      // Successfully installed new version, try gc before return
      if (gc_enabled) {
        gc_lsn_t gc_lsn = gc_get_lsn();
        vid = ver_buf.v.next;
        bool collect = false;
        while (vid != version_id_null) {
          version_read(vid, &ver_buf);
          const version_id_t vid_next = ver_buf.v.next;
          if (!collect) {
            if (ver_buf.v.csn <= gc_lsn) {
              // This vid's begin_csn is older than gc_lsn
              // collect all versions after this
              collect = true;
              // Nullify the chain end
              if (ver_buf.v.next != version_id_null) {
                ver_buf.v.next = version_id_null;
                version_write(vid, &ver_buf);
              }
            }
          }
          else { // collect
            version_free(vid);
          }
          vid = vid_next;
        }
      }
      return STATUS_SUCCESS;
    }
    else {
      // Trying to modify outdated version: conflict
      return STATUS_CONFLICT;
    }
  }
}

void object_finalize(oid_t oid, xid_t xid, csn_t csn, bool commit) {
  __dma_aligned version_t ver_buf;
  version_id_t vid;
  if (!IS_SECONDARY_OID(oid)) { // primary
    vid = oid_get(oid);
    assert_print(vid != version_id_null);
    version_read(vid, &ver_buf);
    assert_print(!ver_buf.v.meta.is_free_slot);
    assert_print(ver_buf.v.meta.dirty && ver_buf.v.csn == xid);
    if (commit) {
      // expose the dirty head
      ver_buf.v.csn = csn;
      ver_buf.v.meta.dirty = false;
      version_write(vid, &ver_buf);
    }
    else { // abort
      // remove the dirty head
      oid_set(oid, ver_buf.v.next);
      version_free(vid);
    }
  }
  else {
    vid = SECONDARY_OID_TO_VID(oid);
    version_read(vid, &ver_buf);
    assert_print(!ver_buf.s.meta.is_free_slot);
    assert_print(ver_buf.s.meta.dirty && ver_buf.s.begin_csn == xid);
    if (commit) {
      ver_buf.s.begin_csn = csn;
      ver_buf.s.meta.dirty = false;
      version_write(vid, &ver_buf);
    }
    else { // abort
      ver_buf.s.begin_csn = (csn_t)-1;
      ver_buf.s.meta.dirty = false;
      version_write(vid, &ver_buf);
    }
  }
}

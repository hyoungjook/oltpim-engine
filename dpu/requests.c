#include <stddef.h>
#include "requests.h"
#include "btree.h"
#include "oid.h"
#include "version.h"

static btree_t main_index;

void process_init_global() {
  btree_init_global();
  btree_init(&main_index, false);
  oid_manager_init_global();
  version_init_global();
}

#define DECLARE_REQUEST_FUNC(_1, name, _2, _3, _4, _5, _6, ...) \
static inline void process_##name(args_##name##_t *args, rets_##name##_t *rets);

REQUEST_TYPES_LIST(DECLARE_REQUEST_FUNC)

#undef DECLARE_REQUEST_FUNC

void process_request(request_type_t request_type, args_any_t *args, rets_any_t *rets) {
  #define case_process_request(name) \
  process_##name(&args->name, &rets->name);

  REQUEST_SWITCH_CASE(request_type, case_process_request)

  #undef case_process_request
}

static_assert(sizeof(btree_val_t) == sizeof(oid_t), "");
static_assert(sizeof(oid_value_t) == sizeof(version_t), "");

static inline void process_insert(args_insert_t *args, rets_insert_t *rets) {
  status_t status = STATUS_SUCCESS;
  // Allocate oid
  oid_t new_oid = oid_alloc_acquire(args->xid);
  // Try insert to btree
  btree_key_t key = *(btree_key_t*)args->key;
  btree_val_t old_oid = btree_insert(main_index, key, new_oid);
  version_t head = version_null;
  if (old_oid != BTREE_NOVAL) {
    // key already exists
    oid_free_release(new_oid);
    head = (version_t)oid_get(old_oid);
    if (head != version_null && version_read(head, args->csn, NULL)) {
      // visible version exists, insert fails
      status = STATUS_FAILED;
      goto end;
    }
    new_oid = old_oid;
    // already deleted version, try insert as an update
    goto install;
  }

  // try insert
  install:
  if (!oid_try_acquire(new_oid, args->xid)) {
    status = STATUS_CONFLICT;
    goto end;
  }
  else {
    version_t new_head;
    if (head != version_null) {
      new_head = version_update(head, args->csn, args->value);
      assert(new_head != head);
    }
    else {
      new_head = version_create(args->csn, args->value);
    }
    oid_set(old_oid, new_head);
  }

  // return
  end:
  rets->status = status;
}

static inline void process_select(args_select_t *args, rets_select_t *rets) {
  status_t status = STATUS_FAILED;
  // query btree
  btree_key_t key = *(btree_key_t*)args->key;
  oid_t oid = btree_get(main_index, key);
  if (oid != BTREE_NOVAL) {
    version_t head = (version_t)oid_get(oid);
    if (head != version_null) {
      version_value_t value;
      if (version_read(head, args->csn, &value)) {
        // Read succeed!
        rets->value = value;
        status = STATUS_SUCCESS;
      }
    }
  }
  // return
  rets->status = status;
}

static inline void process_update(args_update_t *args, rets_update_t *rets) {
  status_t status = STATUS_FAILED;
  // query btree
  btree_key_t key = *(btree_key_t*)args->key;
  oid_t oid = btree_get(main_index, key);
  if (oid != BTREE_NOVAL) {
    if (!oid_try_acquire(oid, args->xid)) {
      status = STATUS_CONFLICT;
      goto end;
    }
    else {
      // oid acquired
      version_t head = (version_t)oid_get(oid);
      if (head != version_null && version_read(head, args->csn, &rets->old_value)) {
        
      }
      if (head == version_null) {
        // Invisible
        status = STATUS_FAILED;
      }
      else {
        version_t new_head = version_update(head, args->csn, args->new_value, &rets->old_value);
        oid_set(oid, new_head);
      }
    }
  }

  end:
  rets->status = status;
}

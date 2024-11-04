#include <stddef.h>
#include "interface.h"
#include "requests.h"
#include "btree.h"
#include "object.h"
#include "wset.h"

static btree_t main_index;

void process_init_global() {
  btree_init_global();
  btree_init(&main_index, false);
  object_init_global();
  wset_init_global();
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

static inline void process_insert(args_insert_t *args, rets_insert_t *rets) {
  status_t status = STATUS_FAILED;
  bool add_to_write_set = true; // default, if btree_insert succeeds
  // Allocate object
  oid_t oid = object_create_acquire(args->xid, args->value);
  // Try insert to btree
  btree_val_t old_oid = btree_insert(main_index, args->key, oid);
  if (old_oid != BTREE_NOVAL) {
    // key already exists
    object_cancel_create(oid);
    oid = old_oid;
    if (!object_read(oid, args->xid, args->csn, NULL)) {
      // already deleted version, try insert as an update
      status = object_update(oid, args->xid, args->csn, args->value,
        NULL, false, &add_to_write_set);
    }
    else {
      // visible version exists
      // status = STATUS_FAILED;
    }
  }
  else {
    // insert succeed
    status = STATUS_SUCCESS;
  }
  if (status == STATUS_SUCCESS && add_to_write_set) {
    wset_add(args->xid, oid);
  }
  // return
  rets->status = status;
}

static inline void process_get(args_get_t *args, rets_get_t *rets) {
  status_t status = STATUS_FAILED;
  // query btree
  oid_t oid = btree_get(main_index, args->key);
  if (oid != BTREE_NOVAL && object_read(oid, args->xid, args->csn, &rets->value)) {
    status = STATUS_SUCCESS;
  }
  // return
  rets->status = status;
}

static inline void process_update(args_update_t *args, rets_update_t *rets) {
  status_t status = STATUS_FAILED;
  bool add_to_write_set = false;
  // query btree
  oid_t oid = btree_get(main_index, args->key);
  if (oid != BTREE_NOVAL) {
    status = object_update(oid, args->xid, args->csn, args->new_value,
      &rets->old_value, false, &add_to_write_set);
  }
  if (status == STATUS_SUCCESS && add_to_write_set) {
    wset_add(args->xid, oid);
  }
  // return
  rets->status = status;
}

static inline void process_remove(args_remove_t *args, rets_remove_t *rets) {
  status_t status = STATUS_FAILED;
  bool add_to_write_set = false;
  // query btree
  oid_t oid = btree_get(main_index, args->key);
  if (oid != BTREE_NOVAL) {
    status = object_update(oid, args->xid, args->csn, 0, NULL, true, &add_to_write_set);
  }
  if (status == STATUS_SUCCESS && add_to_write_set) {
    wset_add(args->xid, oid);
  }
  // return
  rets->status = status;
}

typedef struct _finalize_arg {
  xid_t xid;
  csn_t csn;
  bool commit;
} finalize_arg;
static void finalize_callback(oid_t oid, void *arg) {
  finalize_arg *a = (finalize_arg*)arg;
  object_finalize(oid, a->xid, a->csn, a->commit);
}

static inline void process_commit(args_commit_t *args, rets_commit_t *rets) {
  finalize_arg arg;
  arg.xid = args->xid;
  arg.csn = args->csn;
  arg.commit = true;
  wset_traverse_remove(args->xid, finalize_callback, &arg);
}

static inline void process_abort(args_abort_t *args, rets_abort_t *rets) {
  finalize_arg arg;
  arg.xid = args->xid;
  // csn not used
  arg.commit = false;
  wset_traverse_remove(args->xid, finalize_callback, &arg);
}

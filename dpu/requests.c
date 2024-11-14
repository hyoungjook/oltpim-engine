#include <stddef.h>
#include <attributes.h>
#include <stdio.h>
#include "interface.h"
#include "requests.h"
#include "btree.h"
#include "object.h"
#include "wset.h"
#include "global.h"

#define NUM_INDEXES DPU_NUM_INDEXES_SYMBOL
#define INDEX_INFOS DPU_INDEX_INFOS_SYMBOL

__host uint64_t NUM_INDEXES;
__host index_info INDEX_INFOS[DPU_MAX_NUM_INDEXES];
static btree_t index_trees[DPU_MAX_NUM_INDEXES];

void process_init_global() {
  btree_init_global();
  for (uint32_t index_id = 0; index_id < NUM_INDEXES; ++index_id) {
    // Unique map for primary index, Multi map for secondary index
    const bool allow_duplicates = !INDEX_INFOS[index_id].primary;
    btree_init(&index_trees[index_id], allow_duplicates);
  }
  object_init_global();
  wset_init_global();
}

#define DECLARE_REQUEST_FUNC(_1, name, _2, _3, _4, _5, _6, ...) \
static inline void process_##name(args_##name##_t *args, rets_##name##_t *rets, __mram_ptr uint8_t *mrets);

REQUEST_TYPES_LIST(DECLARE_REQUEST_FUNC)

#undef DECLARE_REQUEST_FUNC

void process_request(request_type_t request_type, args_any_t *args, rets_any_t *rets,
    __mram_ptr uint8_t *mrets) {
  #define case_process_request(name) \
  process_##name(&args->name, &rets->name, mrets);

  REQUEST_SWITCH_CASE(request_type, case_process_request)

  #undef case_process_request
}

static_assert(sizeof(btree_val_t) == sizeof(oid_t), "");

static inline void process_insert(args_insert_t *args, rets_insert_t *rets, __mram_ptr uint8_t *_) {
  assert_print(args->index_id < NUM_INDEXES);
  status_t status = STATUS_FAILED;
  bool add_to_write_set = true; // default, if btree_insert succeeds
  // Allocate object
  oid_t oid = object_create_acquire(args->xid, args->value,
    INDEX_INFOS[args->index_id].primary);
  // Try insert to btree
  btree_val_t old_oid = btree_insert(index_trees[args->index_id], args->key, oid);
  if (old_oid != BTREE_NOVAL) {
    // key already exists
    assert_print(INDEX_INFOS[args->index_id].primary);
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
  rets->oid = oid;
  rets->status = status;
}

static inline void process_get(args_get_t *args, rets_get_t *rets, __mram_ptr uint8_t *_) {
  assert_print(args->index_id < NUM_INDEXES);
  status_t status = STATUS_FAILED;
  // query btree
  oid_t oid;
  if (args->oid_query) {
    assert_print(INDEX_INFOS[args->index_id].primary);
    oid = (oid_t)args->key;
  }
  else {
    oid = btree_get(index_trees[args->index_id], args->key);
  }
  if (oid != BTREE_NOVAL && object_read(oid, args->xid, args->csn, &rets->value)) {
    status = STATUS_SUCCESS;
  }
  // return
  rets->status = status;
}

static inline void process_update(args_update_t *args, rets_update_t *rets, __mram_ptr uint8_t *_) {
  assert_print(args->index_id < NUM_INDEXES);
  assert_print(INDEX_INFOS[args->index_id].primary);
  status_t status = STATUS_FAILED;
  bool add_to_write_set = false;
  // query btree
  oid_t oid = btree_get(index_trees[args->index_id], args->key);
  if (oid != BTREE_NOVAL) {
    status = object_update(oid, args->xid, args->csn, args->new_value,
      &rets->old_value, false, &add_to_write_set);
  }
  if (status == STATUS_SUCCESS && add_to_write_set) {
    wset_add(args->xid, oid);
  }
  // return
  rets->oid = oid;
  rets->status = status;
}

static inline void process_remove(args_remove_t *args, rets_remove_t *rets, __mram_ptr uint8_t *_) {
  assert_print(args->index_id < NUM_INDEXES);
  assert_print(INDEX_INFOS[args->index_id].primary);
  status_t status = STATUS_FAILED;
  bool add_to_write_set = false;
  // query btree
  oid_t oid = btree_get(index_trees[args->index_id], args->key);
  if (oid != BTREE_NOVAL) {
    status = object_update(oid, args->xid, args->csn, 0, NULL, true, &add_to_write_set);
  }
  if (status == STATUS_SUCCESS && add_to_write_set) {
    wset_add(args->xid, oid);
  }
  // return
  rets->oid = oid;
  rets->status = status;
}

typedef struct _scan_callback_args {
  uint64_t xid, csn;        // input, xid and csn
  __mram_ptr uint64_t *output_array;   // output, store values here
  uint8_t max_outs;         // input, max number of outputs
  uint8_t outs;             // output, number of found values
  uint8_t status;           // output, status
} scan_callback_args;
static bool scan_callback(oid_t oid, void *args) {
  scan_callback_args *a = (scan_callback_args*)args;
  __dma_aligned uint64_t value;
  if (object_read(oid, a->xid, a->csn, &value)) {
    // succeed
    mram_write(&value, &a->output_array[a->outs], sizeof(uint64_t));
    a->status = STATUS_SUCCESS;
    ++(a->outs);
    if (a->outs >= a->max_outs) return false; // end of scan
  }
  else {
    // invisible: skip to next item
  }
  return true;
}

static inline void process_scan(args_scan_t *args, rets_scan_t *rets, __mram_ptr uint8_t *mrets) {
  assert_print(args->index_id < NUM_INDEXES);
  assert_print(args->keys[0] <= args->keys[1]);
  scan_callback_args scan_args;
  scan_args.xid = args->xid;
  scan_args.csn = args->csn;
  scan_args.output_array = (__mram_ptr uint64_t*)(mrets + sizeof(rets_scan_t));
  scan_args.max_outs = args->max_outs;
  scan_args.outs = 0;
  scan_args.status = STATUS_FAILED;
  // query btree
  btree_scan(index_trees[args->index_id], (uint64_t*)&args->keys, scan_callback, &scan_args);
  rets->status = scan_args.status;
  rets->outs = scan_args.outs;
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

static inline void process_commit(args_commit_t *args, rets_commit_t *rets, __mram_ptr uint8_t *_) {
  finalize_arg arg;
  arg.xid = args->xid;
  arg.csn = args->csn;
  arg.commit = true;
  wset_traverse_remove(args->xid, finalize_callback, &arg);
}

static inline void process_abort(args_abort_t *args, rets_abort_t *rets, __mram_ptr uint8_t *_) {
  finalize_arg arg;
  arg.xid = args->xid;
  // csn not used
  arg.commit = false;
  wset_traverse_remove(args->xid, finalize_callback, &arg);
}

#include "requests.h"
#include "btree.h"
#include "oid.h"

static btree_t main_index;

void process_init_global() {
  btree_init_global();
  btree_init(&main_index, false);
  oid_manager_init_global();
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
  // Allocate oid
  oid_t new_oid = oid_alloc();
  // Try insert to btree
  btree_key_t key = *(btree_key_t*)args->key;
  btree_val_t old_value = btree_insert(main_index, key, new_oid);
  if (old_value != BTREE_NOVAL) {
    // Insert failed
    oid_free(new_oid);
  }
  else {
    // Insert succeed
    oid_set(new_oid, args->value);
  }
  rets->old_value = old_value;
}

static inline void process_get(args_get_t *args, rets_get_t *rets) {
  btree_key_t key = *(btree_key_t*)args->key;
  oid_t oid = btree_get(main_index, key);
  uint32_t value;
  if (oid == BTREE_NOVAL) {
    // Not exists
    value = BTREE_NOVAL;
  }
  else {
    value = oid_get(oid);
  }
  rets->value = value;
}

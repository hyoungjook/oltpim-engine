#include "requests.h"
#include "btree.c"

static btree_t main_index;

void process_init_global() {
  btree_init_global();
  btree_init(&main_index, false);
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

static inline void process_insert(args_insert_t *args, rets_insert_t *rets) {
  btree_key_t key = *(btree_key_t*)args->key;
  rets->old_value = btree_insert(main_index, key, args->value);
}

static inline void process_get(args_get_t *args, rets_get_t *rets) {
  btree_key_t key = *(btree_key_t*)args->key;
  rets->value = btree_get(main_index, key);
}

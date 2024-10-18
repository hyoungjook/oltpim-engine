#include "requests.h"

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
  // TODO
  rets->old_value = args->value + 7;
}

static inline void process_get(args_get_t *args, rets_get_t *rets) {
  // TODO
  rets->value = args->key[0] + 8;
}

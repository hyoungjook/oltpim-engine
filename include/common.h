#ifndef __OLTPIM_COMMON_H__
#define __OLTPIM_COMMON_H__

#include <stdint.h>
#include <assert.h>

// args and rets
#define DPU_ARGS_SYMBOL global_args_buf_interface_
#define DPU_RETS_SYMBOL global_rets_buf_interface_

#define DPU_BUFFER_SIZE (128 * 1024)

/**
 * Request Type Specification
 * @param type_id id of request type
 * @param name name of request type
 * @param args_struct members of args struct.
 *      If rets_size_supp != 0, the first member should be
 *      uint8_t rets_cnt.
 * @param args_size in plain number, to make sure it's same
 *      on CPU and DPU.
 * @param rets_struct members of rets struct.
 * @param rets_size in plain number, should be multiple of 8
 * @param rets_size_supp adds to rets_size, using second member
 *      in args_struct.
 */
#define REQUEST_TYPES_LIST(_, ...) \
_(0, insert,                       \
    uint32_t key[2];               \
    uint32_t value;                \
  , 12,                            \
    uint32_t old_value;            \
    uint32_t pad;                  \
  , 8, 0, __VA_ARGS__)             \
_(1, get,                          \
    uint32_t key[2];               \
  , 8,                             \
    uint32_t value;                \
    uint32_t pad;                  \
  , 8, 0, __VA_ARGS__)

/* Common request definitions */
#define DECLARE_REQUEST(type_id, name, args_struct, args_size,       \
                        rets_struct, rets_size, rets_size_supp, ...) \
  typedef struct _args_##name##_t {           \
    args_struct                               \
  } args_##name##_t;                          \
  static_assert(sizeof(args_##name##_t) == args_size, ""); \
                                              \
  typedef struct _rets_##name##_t {           \
    rets_struct                               \
  } rets_##name##_t;                          \
  static_assert(sizeof(rets_##name##_t) == rets_size, ""); \
  static_assert(rets_size % 8 == 0, ""); \
  static_assert(rets_size_supp == 0, "");
// TODO rets_size_supp

REQUEST_TYPES_LIST(DECLARE_REQUEST)
#undef DECLARE_REQUEST

typedef enum _request_type_t {
#define ENUM_MEMBERS(type_id, name, _1, _2, _3, _4, _5, ...) \
  request_type_##name = type_id,
REQUEST_TYPES_LIST(ENUM_MEMBERS)
#undef ENUM_MEMBERS
} request_type_t;

typedef union _args_any_t {
#define ARGS_UNION_MEMBERS(_1, name, _2, _3, _4, _5, _6, ...) \
  args_##name##_t name;
REQUEST_TYPES_LIST(ARGS_UNION_MEMBERS)
#undef ARGS_UNION_MEMBERS
} args_any_t;

typedef union _rets_any_t {
#define RETS_UNION_MEMBERS(_1, name, _2, _3, _4, _5, _6, ...) \
  rets_##name##_t name;
REQUEST_TYPES_LIST(RETS_UNION_MEMBERS)
#undef RETS_UNION_MEMBERS
} rets_any_t;


#define _REQUEST_SWITCH_CASE_HELPER(_1, name, _2, _3, _4, _5, _6, user_macro) \
case request_type_##name: { \
  user_macro(name) \
} break;

#define REQUEST_SWITCH_CASE(request_type, user_macro) \
switch (request_type) { \
REQUEST_TYPES_LIST(_REQUEST_SWITCH_CASE_HELPER, user_macro) \
}

#endif

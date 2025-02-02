#ifndef __OLTPIM_COMMON_H__
#define __OLTPIM_COMMON_H__

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

// args and rets
#define DPU_ARGS_SYMBOL global_args_buf_interface_
#define DPU_RETS_SYMBOL global_rets_buf_interface_

#define DPU_BUFFER_SIZE (256 * 1024)

// index initialization
#define DPU_MAX_NUM_INDEXES (64)
#define DPU_NUM_INDEXES_SYMBOL global_num_indexes_
#define DPU_INDEX_INFOS_SYMBOL global_index_infos_
typedef struct _index_info {
  bool primary;
} index_info;
static_assert((sizeof(index_info) * DPU_MAX_NUM_INDEXES) % 8 == 0, "");

// Global configuration
#define DPU_GC_PROB_SYMBOL    global_gc_prob_
#define DPU_GC_PROB_BASE      128

/**
 * Status codes
 */
typedef uint8_t status_t;
#define STATUS_SUCCESS  ((status_t)1)
#define STATUS_FAILED   ((status_t)2)
#define STATUS_CONFLICT ((status_t)3)
#define CHECK_VALID_STATUS(status) \
  assert(((status) >= STATUS_SUCCESS) || \
         ((status) <= STATUS_CONFLICT))

#define REQ_GET_STATUS(value_status) \
  (((value_status) >= (uint64_t)(-3)) ? \
  -(value_status) : STATUS_SUCCESS)

/**
 * Request Type Specification
 * @param type_id id of request type, should be 0, 1, ...
 * @param name name of request type
 * @param priority priority of request, required to separate
 *      btree inserts from btree reads.
 * @param args_struct members of args struct.
 *      If rets_size_supp != 0, the first member should be
 *      uint8_t rets_cnt.
 * @param args_size in plain number, to make sure it's same
 *      on CPU and DPU.
 * @param rets_struct members of rets struct.
 * @param rets_size in plain number, should be multiple of 8
 * @param rets_size_supp adds to rets_size, using second uint8 member
 *      in args_struct. Use "__rn" to refer the second member.
 */
#define REQUEST_TYPES_LIST(_, ...)  \
_(0, insert, 0,                     \
    uint64_t key;                   \
    uint64_t value;                 \
    struct {                        \
      uint64_t xid: 56;             \
      uint8_t index_id;             \
    } xid_s;                        \
    uint64_t csn;                   \
  , 32,                             \
    uint64_t gc_begin;              \
    uint32_t oid;                   \
    uint16_t gc_num;                \
    uint8_t status;                 \
  , 16, 0, __VA_ARGS__)             \
_(1, get, 1,                        \
    uint64_t key;                   \
    struct {                        \
      uint64_t xid: 56;             \
      uint8_t index_id: 7;          \
      uint8_t oid_query: 1;         \
    } xid_s;                        \
    uint64_t csn;                   \
  , 24,                             \
    uint64_t value_status;          \
  , 8, 0, __VA_ARGS__)              \
_(2, update, 1,                     \
    uint64_t key;                   \
    uint64_t new_value;             \
    struct {                        \
      uint64_t xid: 56;             \
      uint8_t index_id;             \
    } xid_s;                        \
    uint64_t csn;                   \
  , 32,                             \
    uint64_t old_value;             \
    uint64_t gc_begin;              \
    uint32_t oid;                   \
    uint16_t gc_num;                \
    uint8_t status;                 \
  , 24, 0, __VA_ARGS__)             \
_(3, remove, 1,                     \
    uint64_t key;                   \
    struct {                        \
      uint64_t xid: 56;             \
      uint8_t index_id;             \
    } xid_s;                        \
    uint64_t csn;                   \
  , 24,                             \
    uint64_t gc_begin;              \
    uint32_t oid;                   \
    uint16_t gc_num;                \
    uint8_t status;                 \
  , 16, 0, __VA_ARGS__)             \
_(4, scan, 1,                       \
    struct {                        \
      uint8_t max_outs;             \
      uint8_t index_id;             \
      uint64_t xid: 48;             \
    } xid_s;                        \
    uint64_t keys[2];               \
    uint64_t csn;                   \
  , 32,                             \
    uint8_t status;                 \
    uint8_t outs;                   \
    uint8_t pad[6];                 \
    uint64_t values[0];             \
  , 8, 8 * (__rn), __VA_ARGS__)     \
_(5, commit, 0,                     \
    uint64_t xid;                   \
    uint64_t csn;                   \
  , 16,                             \
    uint64_t pad;                   \
  , 8, -8, __VA_ARGS__)             \
_(6, abort, 0,                      \
    uint64_t xid;                   \
  , 8,                              \
    uint64_t pad;                   \
  , 8, -8, __VA_ARGS__)             \
_(7, gc, 0,                         \
    uint64_t gc_lsn;                \
  , 8,                              \
    uint64_t pad;                   \
  , 8, -8, __VA_ARGS__)             \
_(8, insertonly, 0,                 \
    uint64_t key;                   \
    uint32_t value;                 \
    uint8_t index_id;               \
  , 16,                             \
    uint8_t status;                 \
    uint8_t pad[7];                 \
  , 8, 0, __VA_ARGS__)              \
_(9, getonly, 1,                    \
    uint64_t key;                   \
    uint8_t index_id;               \
  , 16,                             \
    uint32_t value;                 \
    uint8_t status;                 \
  , 8, 0, __VA_ARGS__)              \

#define NUM_PRIORITIES 2
#define REQUEST_MAX_ARGS_SIZE 32

/* Secondary index value */
#define SVALUE_MAKE(pim_id, oid) (((uint64_t)(pim_id) << 32) | ((uint64_t)(oid)))
#define SVALUE_GET_OID(svalue) (uint32_t)(svalue)
#define SVALUE_GET_PIMID(svalue) (uint16_t)(svalue >> 32)

/* Common request definitions */
#define DECLARE_REQUEST(type_id, name, priority, args_struct, args_size,  \
                        rets_struct, rets_size, rets_size_supp, ...)      \
  static_assert(0 <= priority && priority < NUM_PRIORITIES, ""); \
                                              \
  typedef struct _args_##name##_t {           \
    args_struct                               \
  } args_##name##_t;                          \
  static_assert(sizeof(args_##name##_t) == args_size, ""); \
  static_assert(sizeof(args_##name##_t) <= REQUEST_MAX_ARGS_SIZE, ""); \
                                              \
  typedef struct _rets_##name##_t {           \
    rets_struct                               \
  } rets_##name##_t;                          \
  static_assert(sizeof(rets_##name##_t) == rets_size, ""); \
  static_assert(rets_size % 8 == 0, "");      \
                                              \
  static inline uint32_t req_##name##_rets_size(args_##name##_t *args) { \
    uint8_t __rn = *(uint8_t*)args; (void)__rn; \
    return sizeof(rets_##name##_t) + (rets_size_supp); \
  }                                           \

REQUEST_TYPES_LIST(DECLARE_REQUEST)
#undef DECLARE_REQUEST

typedef enum _request_type_t {
#define ENUM_MEMBERS(type_id, name, ...) \
  request_type_##name = type_id,
REQUEST_TYPES_LIST(ENUM_MEMBERS)
#undef ENUM_MEMBERS
  request_type_priority_separator = 0xFF,
} request_type_t;

typedef union _args_any_t {
#define ARGS_UNION_MEMBERS(_1, name, ...) \
  args_##name##_t name;
REQUEST_TYPES_LIST(ARGS_UNION_MEMBERS)
#undef ARGS_UNION_MEMBERS
} args_any_t;

typedef union _rets_any_t {
#define RETS_UNION_MEMBERS(_1, name, ...) \
  rets_##name##_t name;
REQUEST_TYPES_LIST(RETS_UNION_MEMBERS)
#undef RETS_UNION_MEMBERS
} rets_any_t;


#define _REQUEST_SWITCH_CASE_HELPER(_1, name, _2, _3, _4, _5, _6, _7, user_macro) \
case request_type_##name: { \
  user_macro(name) \
} break;

#define REQUEST_SWITCH_CASE(request_type, user_macro, ...) \
switch (request_type) { \
REQUEST_TYPES_LIST(_REQUEST_SWITCH_CASE_HELPER, user_macro) \
case request_type_priority_separator: { \
  __VA_ARGS__ \
} break; \
}

#endif

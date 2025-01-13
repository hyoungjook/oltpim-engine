#ifndef __OLTPIM_GLOBAL_H__
#define __OLTPIM_GLOBAL_H__

#include <assert.h>
#include <stdio.h>

#ifndef NR_TASKLETS
#define NR_TASKLETS 8
#endif

#ifdef NDEBUG
#define assert_print(expression) ((void)0)
#else
#define assert_print(expression) \
{ \
  if (!(expression)) { \
    printf("fault %s:%u\n", __FILE__, __LINE__);  \
    assert(false);  \
  } \
}
#endif

/** 
 * Some functions should not be inlined to reduce
 * the IRAM footprint.
*/
#define __noinline __attribute__((noinline))

/**
 * MRAM capacity breakdown configuration
 * 
 * The total MRAM capacity is 64MB.
 * Configurations should fit in the capacity while
 * maximizing the fixed-size allocator capacity.
 * 
 */

/* Version objects */
// sizeof(version_t) = 32
// 32 * 2^20 = 32MB
#define VERSION_ALLOCATOR_SIZE_BITS (20)
#define __VERSION_ALLOCATOR_SIZEOF_ENTITY (32) // for debug

/* Btree nodes */
// sizeof(node_t) = 8*DEGREE + 4*(DEGREE+2) = 12*DEGREE+8 = 128
// 128 * 2^17 = 16MB
#define BT_DEGREE (10)
#define BT_MAX_DEPTH (16)
#define BTREE_ALLOCATOR_SIZE_BITS (17)
#define __BTREE_ALLOCATOR_SIZEOF_ENTITY (128) // for debug

/* Linked list nodes */
// sizeof(ws_list_node_t) = 4*LIST_DEGREE + 8 = 64
// 64 * 2^17 = 8MB
#define LIST_DEGREE (14)
#define LIST_ALLOCATOR_SIZE_BITS (17)
#define __LIST_ALLOCATOR_SIZEOF_ENTITY (64) // for debug

/* OID array */
// 4 * 2^20 = 4MB
#define OID_ARRAY_SIZE_BITS (20)

/* XID hashmap */
// sizeof(xid_map_pair_t) = 16
// 16 * 2^17 = 2MB
#define HASHMAP_SIZE_BITS (17)

#endif

#ifndef __OLTPIM_WSET_H__
#define __OLTPIM_WSET_H__

#include <stdint.h>
#include <stdbool.h>
#include "oid.h"
#include "object.h"

/**
 * Per-txn write set.
 */
typedef void (*wset_callback)(oid_t, void*);

/**
 * Initialize the global structrues.
 * Should be called once at the beginning.
 */
void wset_init_global();

/**
 * Add oid to the write set for txn xid.
 * @param xid xid of txn.
 * @param oid oid of updated object
 */
void wset_add(xid_t xid, oid_t oid);

/**
 * Get write set of xid, remove the entry,
 * and traverse it and call func for each oid in the set.
 * @param xid xid.
 * @param callback callback to run for each oid.
 * @param arg arg for the callback.
 */
void wset_traverse_remove(xid_t xid, wset_callback callback, void *arg);

#endif

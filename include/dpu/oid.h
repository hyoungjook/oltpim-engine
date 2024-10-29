#ifndef __OLTPIM_OID_H__
#define __OLTPIM_OID_H__

#include <stdint.h>
#include <stdbool.h>

/**
 * OID Manager.
 */
typedef uint32_t oid_t;
typedef uint32_t oid_value_t;
#define oid_value_null ((oid_value_t)-1)
typedef uint64_t xid_t;

/**
 * Initialize the global structrues.
 * Should be called once at the beginning.
 */
void oid_manager_init_global();

/**
 * Get an available oid and acquire with xid. Initializes value to null.
 * @return oid.
 */
oid_t oid_alloc_acquire(xid_t xid);

/**
 * Free the oid (and implicitly release if acquired)
 * @param oid oid.
 */
void oid_free_release(oid_t oid);

/**
 * Try acquire the oid using xid
 * @param oid oid.
 * @param xid xid.
 * @return true if succeed. false if already acquired.
 */
bool oid_try_acquire(oid_t oid, xid_t xid);

/**
 * Check if the oid is already acquired by other transaction.
 * @param oid oid.
 * @param xid xid.
 * @return true if oid acquired by other xid. false if it's 
 *  free or acquired by me.
 */
bool oid_is_acquired_by_other(oid_t oid, xid_t xid);

/**
 * Release the oid
 * @param oid oid.
 */
void oid_release(oid_t oid);

/**
 * Set value of the oid.
 */
void oid_set(oid_t oid, oid_value_t val);

/**
 * Get value of the oid.
 */
oid_value_t oid_get(oid_t oid);

#endif

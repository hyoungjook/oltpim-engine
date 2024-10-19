#ifndef __OLTPIM_OID_H__
#define __OLTPIM_OID_H__

#include <stdint.h>

/**
 * OID Manager.
 */
typedef uint32_t oid_t;
typedef uint32_t oid_value_t;

/**
 * Initialize the global structrues.
 * Should be called once at the beginning.
 */
void oid_manager_init_global();

/**
 * Get an available oid
 * @return oid.
 */
oid_t oid_alloc();

/**
 * Free the oid
 * @param oid oid.
 */
void oid_free(oid_t oid);

/**
 * Set value of the oid.
 */
void oid_set(oid_t oid, oid_value_t val);

/**
 * Get value of the oid.
 */
oid_value_t oid_get(oid_t oid);

#endif

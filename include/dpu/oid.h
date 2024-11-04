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

/**
 * Initialize the global structrues.
 * Should be called once at the beginning.
 */
void oid_manager_init_global();

/**
 * Get an available oid and initialize the value.
 * @param val value to initialize.
 * @return allocated oid.
 */
oid_t oid_alloc_set(oid_value_t val);

/**
 * Free the oid.
 * @param oid oid.
 */
void oid_free(oid_t oid);

/**
 * Get value of the oid.
 * @param oid oid.
 * @return value of the oid.
 */
oid_value_t oid_get(oid_t oid);

/**
 * Set value of the oid.
 * @param oid oid.
 * @param val value to set.
 */
void oid_set(oid_t oid, oid_value_t val);

/**
 * Set value of the oid to new_val only if its value is same as old_val.
 * @param oid oid
 * @param old_val old value, input and output
 * @param new_val new value
 * @return whether the value is exchanged
 */
bool oid_compare_exchange(oid_t oid, oid_value_t *old_val, oid_value_t new_val);

#endif

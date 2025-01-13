#ifndef __OLTPIM_OBJECT_H__
#define __OLTPIM_OBJECT_H__

#include <stdint.h>
#include <stdbool.h>
#include "oid.h"
#include "interface.h"

/**
 * Object is a version chain, Newest-to-Oldest.
 */
typedef uint64_t object_value_t;
typedef uint64_t xid_t;
typedef uint64_t csn_t;
#define object_value_null ((object_value_t)-1)

/**
 * Initialize the global structures.
 * Should be called once at the beginning.
 */
void object_init_global();

/**
 * !!!!! Note that although we use the same "oid_t",
 * primary indexes stores oid managed by the oid manager
 * whereas secondary indexes' oids are irrelevant from the
 * primary indexes' oid.
 * It's just the internal pointer to the version data.
 */

/**
 * Create an object and acquire it.
 * @param xid xid of the creator txn.
 * @param new_value initial value for the object.
 * @param primary for primary index, or for secondary index
 * @return oid of the new object.
 */
oid_t object_create_acquire(xid_t xid, object_value_t new_value, bool primary);

/**
 * Call right after object_create, cancels the creation.
 * Assume the oid is not exposed to the outside world.
 * ONLY FOR PRIMARY OID.
 * @param oid oid.
 */
void object_cancel_create(oid_t oid);

/**
 * Read the value of the visible version in the object.
 * @param oid oid.
 * @param xid xid of the reader txn.
 * @param csn begin_csn of the reader txn.
 * @param value output value
 * @return true if visible version exists.
 */
bool object_read(oid_t oid, xid_t xid, csn_t csn, object_value_t *value);

/**
 * Update the object.
 * ONLY FOR PRIMARY OID.
 * @param oid oid.
 * @param xid xid of the updater txn.
 * @param csn begin_csn of the updater txn.
 * @param new_value new value to update.
 * @param old_value output, old value before update
 * @param remove remove instead. ignore new_ & old_value arguments.
 * @param add_to_write_set output, whether it's first time updating this obj
 *  in this txn. only set if succeed.
 * @param gc_begin store the newest obsolete value if gc occurred.
 * @param gc_num store the number of obsolete values if gc occurred.
 *  assumes it's already set to 0 before calling.
 * @return status: succeed / failed / conflict
 */
status_t object_update(oid_t oid, xid_t xid, csn_t csn, object_value_t new_value,
  object_value_t *old_value, bool remove, bool *add_to_write_set,
  object_value_t *gc_begin, uint16_t *gc_num);

/**
 * Finalize the object, on either commit or abort.
 * @param oid oid.
 * @param xid xid of the updater txn.
 * @param csn end_csn of the updater txn.
 * @param commit true if commit, false if abort
 */
void object_finalize(oid_t oid, xid_t xid, csn_t csn, bool commit);

#endif

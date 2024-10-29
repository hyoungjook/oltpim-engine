#ifndef __OLTPIM_VERSION_H__
#define __OLTPIM_VERSION_H__

#include <stdint.h>
#include <stdbool.h>

/**
 * Version chain, Newest-to-Oldest.
 */
typedef uint32_t version_t;
typedef uint64_t csn_t;
typedef uint64_t version_value_t;
#define version_null ((version_t)-1)
#define version_value_null ((version_value_t)-1)

/**
 * Initialize the global structures.
 * Should be called once at the beginning.
 */
void version_init_global();

/**
 * Create a version chain with one version object.
 * @param csn begin_csn of the transaction.
 * @param new_value value of the new version.
 * @return version chain.
 */
version_t version_create(csn_t csn, version_value_t new_value);

/**
 * Traverse the version chain.
 * @param csn begin_csn of the reader transaction.
 * @param ver version chain.
 * @param value output value
 * @return true if visible version exists.
 */
bool version_read(version_t ver, csn_t csn, version_value_t *value);

/**
 * Update the version chain. Assume its oid is already acquired and
 * version_read(ver) already succeed so the update always succeeds. 
 * @param ver version chain.
 * @param csn begin_csn of the updater transaction.
 * @param new_value new value to add to the new version.
 * @return the updated version chain.
 */
version_t version_update(version_t ver, csn_t csn, version_value_t new_value);

/**
 * Finalize the updated version chain, on either commit or abort.
 * @param ver version chain.
 * @param csn end_csn of the updater transaction.
 * @param commit true if commit, false if abort
 * @return the updated version chain. Can return version_null on abort.
 */
version_t version_finalize(version_t ver, csn_t csn, bool commit);

#endif

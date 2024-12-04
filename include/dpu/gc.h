#ifndef __OLTPIM_GC_H__
#define __OLTPIM_GC_H__

#include <stdint.h>
#include <stdbool.h>

typedef uint64_t gc_lsn_t;

/**
 * Initialize the global structures.
 * Should be called once at the beginning.
 */
void gc_init_global();

/**
 * Is GC enabled?
 */
extern bool gc_enabled;

/**
 * Update the gc lsn.
 * @param gc_lsn new gc_lsn.
 */
void gc_update_lsn(gc_lsn_t gc_lsn);

/**
 * Get the gc lsn.
 * @return current gc_lsn.
 */
gc_lsn_t gc_get_lsn();

#endif

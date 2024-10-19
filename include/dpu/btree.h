#ifndef __OLTPIM_BTREE_H__
#define __OLTPIM_BTREE_H__

#include <stdbool.h>
#include <stdint.h>

/**
 * A concurrent B+ tree implementation. 
 */
typedef void *btree_t;
typedef uint64_t btree_key_t;
typedef uintptr_t btree_val_t;
#define BTREE_NOVAL ((btree_val_t)-1)

/**
 * Intialize the global structures.
 * Should be called once at the beginning.
 */
void btree_init_global();

/**
 * Initialize a btree.
 * @param allow_duplicates whether it's multimap or map
 */
void btree_init(btree_t *bt, bool allow_duplicates);

/**
 * Get the value of the given key.
 * @param bt btree structure.
 * @param key Key to search.
 * @return The value for the key stored in the btree, or BTREE_NOVAL if not.
 */
btree_val_t btree_get(btree_t bt, btree_key_t key);

/**
 * Scan the index.
 * @param bt btree structure.
 * @param range if false, get all values with key[0]. (allow_duplicates = true)
 *              if true, scan (key[0], key[1]).
 * @param max_outs max number of outputs.
 * @param out_vals output stored in this array.
 * @return number of found values.
 */
uint32_t btree_scan(btree_t bt, bool range, btree_key_t *key,
  uint32_t max_outs, btree_val_t *out_vals);

/**
 * Insert the (key, value) pair.
 * @param bt btree structure.
 * @param key Key to insert.
 * @param val Value to insert.
 * @return BTREE_NOVAL if insert succeeds.
 *    If !allow_duplicate and key already exists, return the value.
 */
btree_val_t btree_insert(btree_t bt, btree_key_t key, btree_val_t val);

#ifdef btree_debug
void btree_print(btree_t bt);
#endif

#endif

#ifndef __OLTPIM_REQUESTS_H__
#define __OLTPIM_REQUESTS_H__

#include "interface.h"
#include <mram.h>

void process_init_global();

/**
 * Process the request.
 * @param request_type request type.
 * @param args wram pointer to the args
 * @param mrets mram pointer to the rets.
 *    This function is responsible to copy both the fixed-lenght part
 *    and the variable-length part of the return value to the mram.
 *    Both offset and size is guaranteed to be 8-bytes aligned.
 */
void process_request(request_type_t request_type, args_any_t *args,
  __mram_ptr uint8_t *mrets);

#endif

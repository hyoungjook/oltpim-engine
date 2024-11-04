#ifndef __OLTPIM_REQUESTS_H__
#define __OLTPIM_REQUESTS_H__

#include "interface.h"
#include <mram.h>

void process_init_global();

/**
 * Process the request.
 * @param request_type request type.
 * @param args wram pointer to the args
 * @param rets wram pointer to the rets
 * @param mrets supplementary mram pointer to the rets. This points to the
 *    mram location that is copied to wram buffer rets. It is only used to
 *    store the variable-length return value. The fixed-length part is
 *    copied back in the main() function.
 */
void process_request(request_type_t request_type, args_any_t *args, rets_any_t *rets,
  __mram_ptr uint8_t *mrets);

#endif

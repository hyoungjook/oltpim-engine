#ifndef __OLTPIM_REQUESTS_H__
#define __OLTPIM_REQUESTS_H__

#include "interface.h"

void process_init_global();

void process_request(request_type_t request_type, args_any_t *args, rets_any_t *rets);

#endif

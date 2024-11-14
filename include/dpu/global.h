#ifndef __OLTPIM_GLOBAL_H__
#define __OLTPIM_GLOBAL_H__

#include <assert.h>
#include <stdio.h>

#ifndef NR_TASKLETS
#define NR_TASKLETS 8
#endif

#ifdef NDEBUG
#define assert_print(expression) ((void)0)
#else
#define assert_print(expression) \
{ \
  if (!(expression)) { \
    printf("fault %s:%u\n", __FILE__, __LINE__);  \
    assert(false);  \
  } \
}
#endif

#endif

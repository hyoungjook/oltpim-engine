#pragma once
#include "interface.h"

// C++-only, variable-length rets struct extension

template <int rn>
struct rets_scan_ext {
    rets_scan_t base;
    uint64_t values[rn];
};

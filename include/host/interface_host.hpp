#pragma once
#include "interface.h"
#include "engine.hpp"

// C++-only, variable-length rets struct extension
template <int rn>
struct rets_scan_ext {
  rets_scan_t base;
  uint64_t values[rn];
};

// alias for request<> structs
namespace oltpim {

using request_insert = request<request_type_insert, args_insert_t, rets_insert_t>;
using request_get = request<request_type_get, args_get_t, rets_get_t>;
using request_update = request<request_type_update, args_update_t, rets_update_t>;
using request_remove = request<request_type_remove, args_remove_t, rets_remove_t>;
using request_commit = request_norets<request_type_commit, args_commit_t>;
using request_abort = request_norets<request_type_abort, args_abort_t>;

template <int rn>
struct request_scan {
  using t = request<request_type_scan, args_scan_t, rets_scan_ext<rn>>;
};

}

#pragma once
#include <stdio.h>
#include <stdlib.h>

namespace oltpim {
// Constants
#define CACHE_LINE 64

// Macros
#ifdef NDEBUG
#define OLTPIM_ASSERT(stmt) (void)(stmt)
#else
#define OLTPIM_ASSERT(stmt) \
do { \
  bool _status = stmt; \
  if (!_status) { \
    fprintf(stderr, "Assertion failed: %s (%s:%d)\n", \
      #stmt, __FILE__, __LINE__); \
    exit(1); \
  } \
} while (0)
#endif

#define __STRINGIFY(x) #x
#define TOSTRING(x) __STRINGIFY(x)

#define ALIGN8(x) (((x) + 7) & (~7))

// Structures
template <class T>
class array {
 public:
  array() {}
  array(size_t num_elems) {
    alloc(num_elems);
  }
  void alloc(size_t num_elems) {
    _num_elems = num_elems;
    _arr = (T*)malloc(sizeof(T) * num_elems);
    for (size_t each_elem = 0; each_elem < num_elems; ++each_elem) {
      new (&_arr[each_elem]) T();
    }
  }
  array(array&& other) {
    _num_elems = other._num_elems;
    _arr = other._arr;
    other._arr = nullptr;
  }
  ~array() {
    if (_arr) {
      for (size_t each_elem = 0; each_elem < _num_elems; ++each_elem) {
        _arr[each_elem].~T();
      }
      free(_arr);
    }
  }

  inline T& operator[](size_t idx) {return _arr[idx];}
  inline const T& operator[](size_t idx) const {return _arr[idx];}
  inline size_t size() {return _num_elems;}
 private:
  size_t _num_elems;
  T *_arr = nullptr;
};

}

#pragma once

/**
 * The UPMEM direct interface modified from 
 * https://github.com/Loremkang/upmem-sdk-light/blob/7ca4de5b75fc0781321c8557815ffe3e247ffc18/src/pim_interface/direct_interface.hpp
 * It allows CPU-PIM copy to occur entirely within the calling thread
 */

#include <immintrin.h>
#include <x86intrin.h>
#include <cinttypes>
#include <assert.h>
#include <libudev.h>

extern "C" {
#include <dpu.h>
#include <dpu_description.h>
#include <dpu_management.h>
#include <dpu_memory.h>
#include <dpu_target.h>

/**
 * Non-exposed internal structs, specific to sdk 2024.2.0
 * The purpose is to extract ptr_region from the rank
 */
struct dpu_rank_udev {
    struct udev *udev;
    struct udev_device *dev;
    struct udev_enumerate *enumerate;
    struct udev_list_entry *devices;
};

struct dpu_rank_fs {
    char rank_path[128];
    int fd_rank;
    int fd_dax;
    struct dpu_rank_udev udev, udev_dax, udev_parent;
};

struct dpu_transfer_thread_configuration {
    uint32_t nb_thread_per_pool;
    uint32_t threshold_1_thread;
    uint32_t threshold_2_threads;
    uint32_t threshold_4_threads;
};

struct dpu_region_address_translation {
    struct dpu_hw_description_t *desc;
    uint8_t backend_id;
    uint64_t capabilities;
    uint64_t hybrid_mmap_size;
    struct dpu_transfer_thread_configuration xfer_thread_conf;
    bool one_read;
    void *private_;
    void *init_rank;
    void *destroy_rank;
    void *write_to_rank;
    void *read_from_rank;
    void *write_to_cis;
    void *read_from_cis;
};

typedef struct _hw_dpu_rank_allocation_parameters_t {
    struct dpu_rank_fs rank_fs;
    struct dpu_region_address_translation translate;
    uint64_t region_size;
    uint8_t mode, dpu_chip_id, backend_id;
    uint8_t channel_id;
    uint8_t *ptr_region;
    bool bypass_module_compatibility;
} *hw_dpu_rank_allocation_parameters_t;

struct dpu_rank_t {
    dpu_type_t type;
    dpu_rank_id_t rank_id;
    dpu_rank_id_t rank_handler_allocator_id;
    dpu_description_t description;
};

/**
 * This is declared with __API_SYMBOL__ but not declared in header
 */
dpu_error_t dpu_switch_mux_for_rank(struct dpu_rank_t *rank,
				    bool set_mux_for_host);

}

namespace upmem {

class direct {
private:
  static void byte_interleave_avx512(uint64_t *input, uint64_t *output, bool use_stream) {
    __m512i load = _mm512_loadu_si512(input);
    // LEVEL 0
    __m512i vindex = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    __m512i gathered = _mm512_permutexvar_epi32(vindex, load);
    // LEVEL 1
    __m512i mask = _mm512_set_epi64(0x0f0b07030e0a0602ULL,
      0x0d0905010c080400ULL,
      0x0f0b07030e0a0602ULL,
      0x0d0905010c080400ULL,
      0x0f0b07030e0a0602ULL,
      0x0d0905010c080400ULL,
      0x0f0b07030e0a0602ULL,
      0x0d0905010c080400ULL);
    __m512i transpose = _mm512_shuffle_epi8(gathered, mask);
    // LEVEL 2
    __m512i perm = _mm512_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
    __m512i final = _mm512_permutexvar_epi32(perm, transpose);
    if (use_stream) {
        _mm512_stream_si512((__m512i *)output, final);
        return;
    }
    _mm512_storeu_si512((__m512i *)output, final);
  }

  static inline uint64_t GetCorrectOffsetMRAM(uint64_t address_offset, uint32_t dpu_id) {
    uint64_t mask_move_7 = (~((1 << 22) - 1)) + (1 << 13);  // 31..22, 13
    uint64_t mask_move_6 = ((1 << 22) - (1 << 15));         // 21..15
    uint64_t mask_move_14 = (1 << 14);                      // 14
    uint64_t mask_move_4 = (1 << 13) - 1;                   // 12 .. 0
    return ((address_offset & mask_move_7) << 7) |
           ((address_offset & mask_move_6) << 6) |
           ((address_offset & mask_move_14) << 14) |
           ((address_offset & mask_move_4) << 4) | (dpu_id << 18);
  }

  static inline void SendToRankMRAM(
      uint8_t **buffers, uint32_t symbol_offset, uint8_t *ptr_dest, uint32_t length) {
    assert(symbol_offset % sizeof(uint64_t) == 0);
    assert(length % sizeof(uint64_t) == 0);
    uint64_t cache_line[8];
    for (uint32_t dpu_id = 0; dpu_id < 4; ++dpu_id) {
      for (uint32_t i = 0; i < length / sizeof(uint64_t); ++i) {
        if ((i % 8 == 0) && (i + 8 < length / sizeof(uint64_t))) {
          for (int j = 0; j < 16; j++) {
            __builtin_prefetch(
              ((uint64_t *)buffers[j * 4 + dpu_id]) + i + 8);
          }
        }
        uint64_t offset = GetCorrectOffsetMRAM(symbol_offset + (i * 8), dpu_id);
        for (int j = 0; j < 8; j++) {
            if (buffers[j * 8 + dpu_id] == nullptr) {
                continue;
            }
            cache_line[j] = *(((uint64_t *)buffers[j * 8 + dpu_id]) + i);
        }
        byte_interleave_avx512(cache_line, (uint64_t *)(ptr_dest + offset), true);
        offset += 0x40;
        for (int j = 0; j < 8; j++) {
            if (buffers[j * 8 + dpu_id + 4] == nullptr) {
                continue;
            }
            cache_line[j] = *(((uint64_t *)buffers[j * 8 + dpu_id + 4]) + i);
        }
        byte_interleave_avx512(cache_line, (uint64_t *)(ptr_dest + offset), true);
      }
    }
    __builtin_ia32_mfence();
  }

  static inline void ReceiveFromRankMRAM(
      uint8_t **buffers, uint32_t symbol_offset, uint8_t *ptr_dest, uint32_t length) {
    assert(symbol_offset % sizeof(uint64_t) == 0);
    assert(length % sizeof(uint64_t) == 0);
    for (uint32_t dpu_id = 0; dpu_id < 4; ++dpu_id) {
      for (uint32_t i = 0; i < length / sizeof(uint64_t); ++i) {
        // 8 shards of DPUs
        uint64_t offset = GetCorrectOffsetMRAM(symbol_offset + (i * 8), dpu_id);
        __builtin_ia32_clflushopt((void *)(ptr_dest + offset));
        offset += 0x40;
        __builtin_ia32_clflushopt((void *)(ptr_dest + offset));
      }
    }
    __builtin_ia32_mfence();
    uint64_t cache_line[8], cache_line_interleave[8];
    auto LoadData = [](uint64_t *cache_line, uint8_t *ptr_dest) {
        cache_line[0] = *((volatile uint64_t *)((uint8_t *)ptr_dest +
                                                0 * sizeof(uint64_t)));
        cache_line[1] = *((volatile uint64_t *)((uint8_t *)ptr_dest +
                                                1 * sizeof(uint64_t)));
        cache_line[2] = *((volatile uint64_t *)((uint8_t *)ptr_dest +
                                                2 * sizeof(uint64_t)));
        cache_line[3] = *((volatile uint64_t *)((uint8_t *)ptr_dest +
                                                3 * sizeof(uint64_t)));
        cache_line[4] = *((volatile uint64_t *)((uint8_t *)ptr_dest +
                                                4 * sizeof(uint64_t)));
        cache_line[5] = *((volatile uint64_t *)((uint8_t *)ptr_dest +
                                                5 * sizeof(uint64_t)));
        cache_line[6] = *((volatile uint64_t *)((uint8_t *)ptr_dest +
                                                6 * sizeof(uint64_t)));
        cache_line[7] = *((volatile uint64_t *)((uint8_t *)ptr_dest +
                                                7 * sizeof(uint64_t)));
    };
    for (uint32_t dpu_id = 0; dpu_id < 4; ++dpu_id) {
      for (uint32_t i = 0; i < length / sizeof(uint64_t); ++i) {
        if ((i % 8 == 0) && (i + 8 < length / sizeof(uint64_t))) {
          for (int j = 0; j < 16; j++) {
            __builtin_prefetch(((uint64_t *)buffers[j * 4 + dpu_id]) + i + 8);
          }
        }
        uint64_t offset = GetCorrectOffsetMRAM(symbol_offset + (i * 8), dpu_id);
        if (i + 3 < length / sizeof(uint64_t)) {
          uint64_t offset_prefetch = GetCorrectOffsetMRAM(symbol_offset + ((i + 3) * 8), dpu_id);
          __builtin_prefetch(ptr_dest + offset_prefetch);
          __builtin_prefetch(ptr_dest + offset_prefetch + 0x40);
        }
        // __builtin_prefetch(ptr_dest + offset + 0x40 * 6);
        // __builtin_prefetch(ptr_dest + offset + 0x40 * 7);
        LoadData(cache_line, ptr_dest + offset);
        byte_interleave_avx512(cache_line, cache_line_interleave, false);
        for (int j = 0; j < 8; j++) {
          if (buffers[j * 8 + dpu_id] == nullptr) {
              continue;
          }
          *(((uint64_t *)buffers[j * 8 + dpu_id]) + i) = cache_line_interleave[j];
        }
        offset += 0x40;
        LoadData(cache_line, ptr_dest + offset);
        byte_interleave_avx512(cache_line, cache_line_interleave, false);
        for (int j = 0; j < 8; j++) {
          if (buffers[j * 8 + dpu_id + 4] == nullptr) {
              continue;
          }
          *(((uint64_t *)buffers[j * 8 + dpu_id + 4]) + i) = cache_line_interleave[j];
        }
      }
    }
    for (uint32_t dpu_id = 0; dpu_id < 4; ++dpu_id) {
      for (uint32_t i = 0; i < length / sizeof(uint64_t); ++i) {
        // 8 shards of DPUs
        uint64_t offset = GetCorrectOffsetMRAM(symbol_offset + (i * 8), dpu_id);
        __builtin_ia32_clflushopt((void *)(ptr_dest + offset));
        offset += 0x40;
        __builtin_ia32_clflushopt((void *)(ptr_dest + offset));
      }
    }
    __builtin_ia32_mfence();
  }

public:
  static dpu_error_t copy_to_mrams(dpu_rank_t *rank, dpu_transfer_matrix *matrix) {
    DPU_ASSERT(dpu_switch_mux_for_rank(rank, true));
    dpu_description_t desc = rank->description;
    auto params = (hw_dpu_rank_allocation_parameters_t)(desc->_internals.data);
    uint8_t *base_addr = params->ptr_region;
    SendToRankMRAM((uint8_t**)&matrix->ptr[0], matrix->offset, base_addr, matrix->size);
    return DPU_OK;
  }

  static dpu_error_t copy_from_mrams(dpu_rank_t *rank, dpu_transfer_matrix *matrix) {
    DPU_ASSERT(dpu_switch_mux_for_rank(rank, true));
    dpu_description_t desc = rank->description;
    auto params = (hw_dpu_rank_allocation_parameters_t)(desc->_internals.data);
    uint8_t *base_addr = params->ptr_region;
    ReceiveFromRankMRAM((uint8_t**)&matrix->ptr[0], matrix->offset, base_addr, matrix->size);
    return DPU_OK;
  }

};

}

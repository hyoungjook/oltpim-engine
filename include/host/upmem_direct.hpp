#pragma once

#include <immintrin.h>
#include <x86intrin.h>
#include <cinttypes>
#include <assert.h>
#include <libudev.h>
#include <iostream>

#define UPMEM_USE_DIRECT_MUX

extern "C" {
/* Includes from UPMEM Host APIs and Low-Level APIs */
#include <dpu.h>
#include <dpu_description.h>
#include <dpu_hw_description.h>
#include <dpu_management.h>
#include <dpu_memory.h>
#include <dpu_runner.h>
#include <dpu_target.h>

/* Includes from UPMEM-SDK submodule */
#include "dpu_rank.h"
#include "ufi.h"
#include "ufi_config.h"

// hw/src/rank/hw_dpu_rank.c
// hw/src/commons/dpu_region_address_translation.h
#include "hw_dpu_sysfs.h"
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
    int *init_rank;
    void *destroy_rank;
    void *write_to_rank;
    void *read_from_rank;
    void *write_to_cis;
    void *read_from_cis;
#ifdef __KERNEL__
    int *mmap_hybrid;
#endif
};
typedef struct _fpga_allocation_parameters_t {
  bool activate_ila;
  bool activate_filtering_ila;
  bool activate_mram_bypass;
  bool activate_mram_refresh_emulation;
  unsigned int mram_refresh_emulation_period;
  char *report_path;
  bool cycle_accurate;
} fpga_allocation_parameters_t;
typedef struct _hw_dpu_rank_allocation_parameters_t {
  struct dpu_rank_fs rank_fs;
  struct dpu_region_address_translation translate;
  uint64_t region_size;
  uint8_t mode, dpu_chip_id, backend_id;
  uint8_t channel_id;
  uint8_t *ptr_region;
  bool bypass_module_compatibility;
  /* Backends specific */
  fpga_allocation_parameters_t fpga;
} * hw_dpu_rank_allocation_parameters_t;
}

namespace upmem {

/**
 * The optimized mux_switch API.
 * The main contribution is that we removed unnecessary call to
 * ufi_select_dpu_even_disabled() inside dpu_check_wavegen_mux_status_for_rank().
 * !!!!! ASSUMES ALL DPUS ARE ENABLED !!!!!
 * If not, it may fail. (not tested)
 */
class mux {
public:
  static bool is_switch_required(dpu_rank_t *rank, bool mux_for_host) {
    dpu_description_t desc = rank->description;
    const uint8_t nr_cis = desc->hw.topology.nr_of_control_interfaces;
    const uint8_t nr_dpus_per_ci = desc->hw.topology.nr_of_dpus_per_control_interface;
    for (uint8_t each_slice = 0; each_slice < nr_cis; ++each_slice) {
      dpu_bitfield_t host_mux_mram_state =
        rank->runtime.control_interface.slice_info[each_slice].host_mux_mram_state;
      if ((mux_for_host && __builtin_popcount(host_mux_mram_state) < nr_dpus_per_ci) ||
          (!mux_for_host && host_mux_mram_state)) {
        return true;
      }
    }
    return false;
  }

  static void switch_begin(dpu_rank_t *rank, bool mux_for_host) {
    dpu_description_t desc = rank->description;
    const uint8_t nr_cis = desc->hw.topology.nr_of_control_interfaces;
    const uint8_t nr_dpus_per_ci = desc->hw.topology.nr_of_dpus_per_control_interface;
    for (uint8_t each_slice = 0; each_slice < nr_cis; ++each_slice) {
      rank->runtime.control_interface.slice_info[each_slice].host_mux_mram_state =
        (mux_for_host ? ((1 << nr_dpus_per_ci) - 1) : 0x0);
    }
    uint8_t ci_mask = ALL_CIS;
    DPU_ASSERT((dpu_error_t)ufi_select_all_even_disabled(rank, &ci_mask));
    assert(ci_mask == ALL_CIS);
    DPU_ASSERT((dpu_error_t)ufi_set_mram_mux(rank, ci_mask, mux_for_host ? 0xFF : 0x00));
  }

  static void switch_sync(dpu_rank_t *rank, bool mux_for_host) {
    dpu_description_t desc = rank->description;
    const uint8_t nr_cis = desc->hw.topology.nr_of_control_interfaces;
    const uint8_t nr_dpus_per_ci = desc->hw.topology.nr_of_dpus_per_control_interface;
    uint8_t ci_mask = ALL_CIS;
    // dpu_check_wavegen_mux_status_for_rank
    DPU_ASSERT((dpu_error_t)ufi_write_dma_ctrl(rank, ci_mask, 0xFF, 0x02));
    DPU_ASSERT((dpu_error_t)ufi_clear_dma_ctrl(rank, ci_mask));
    uint8_t result_array[DPU_MAX_NR_CIS];
    const uint8_t wavegen_expected = mux_for_host ? 0x00 : ((1 << 0) | (1 << 1));
    for (uint8_t each_dpu = 0; each_dpu < nr_dpus_per_ci; ++each_dpu) {
      uint32_t timeout = 100;
      bool should_retry = false;
      do {
        DPU_ASSERT((dpu_error_t)ufi_read_dma_ctrl(rank, ci_mask, result_array));
        for (uint8_t each_slice = 0; each_slice < nr_cis; ++each_slice) {
          if ((result_array[each_slice] & 0x7B) != wavegen_expected) {
            should_retry = true;
            break;
          }
        }
        --timeout;
      } while (timeout && should_retry);
      if (!timeout) {
        std::cerr << "upmem::mux::switch_sync: Timeout waiting for result to be correct\n";
        std::abort();
      }
    }
  }

  static void switch_rank(dpu_rank_t *rank, bool mux_for_host) {
#if defined(UPMEM_USE_DIRECT_MUX)
    if (!is_switch_required(rank, mux_for_host)) return;
    switch_begin(rank, mux_for_host);
    switch_sync(rank, mux_for_host);
#else
    DPU_ASSERT(dpu_switch_mux_for_rank(rank, mux_for_host));
#endif
  }
};

class direct_launch {
public:
  static void boot(dpu_rank_t *rank) {
    mux::switch_rank(rank, false);
    // DPU_ASSERT(dpu_boot_rank(rank));
    uint8_t ci_mask = ALL_CIS;
    DPU_ASSERT((dpu_error_t)ufi_select_all(rank, &ci_mask));
    DPU_ASSERT((dpu_error_t)ufi_thread_boot(rank, ci_mask, DPU_BOOT_THREAD, NULL));
  }

  static void poll_status(dpu_rank_t *rank, bool *done, bool *fault) {
    //DPU_ASSERT(dpu_poll_rank(rank));
    //DPU_ASSERT(dpu_status_rank(rank, done, fault));
    uint8_t ci_mask = ALL_CIS;
    dpu_bitfield_t poll_running[DPU_MAX_NR_CIS], poll_fault[DPU_MAX_NR_CIS];
    DPU_ASSERT((dpu_error_t)ufi_select_all(rank, &ci_mask));
    DPU_ASSERT((dpu_error_t)ufi_read_dpu_run(rank, ci_mask, poll_running));
    DPU_ASSERT((dpu_error_t)ufi_read_dpu_fault(rank, ci_mask, poll_fault));
    const uint8_t nr_cis = rank->description->hw.topology.nr_of_control_interfaces;
    *done = true;
    *fault = false;
    for (uint8_t each_ci = 0; each_ci < nr_cis; ++each_ci) {
      dpu_selected_mask_t mask_all = rank->runtime.control_interface.slice_info[each_ci].enabled_dpus;
      dpu_bitfield_t ci_fault = poll_fault[each_ci] & mask_all;
      dpu_bitfield_t ci_running = (poll_running[each_ci] & mask_all) & (~ci_fault);
      *done = *done && (ci_running == 0);
      *fault = *fault || (ci_fault != 0);
    }
  }
};

/**
 * The UPMEM direct interface modified from 
 * https://github.com/Loremkang/upmem-sdk-light/blob/7ca4de5b75fc0781321c8557815ffe3e247ffc18/src/pim_interface/direct_interface.hpp
 * It allows CPU-PIM copy to occur entirely within the calling thread
 */
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
    mux::switch_rank(rank, true);
    dpu_description_t desc = rank->description;
    auto params = (hw_dpu_rank_allocation_parameters_t)(desc->_internals.data);
    uint8_t *base_addr = params->ptr_region;
    SendToRankMRAM((uint8_t**)&matrix->ptr[0], matrix->offset, base_addr, matrix->size);
    return DPU_OK;
  }

  static dpu_error_t copy_from_mrams(dpu_rank_t *rank, dpu_transfer_matrix *matrix) {
    mux::switch_rank(rank, true);
    dpu_description_t desc = rank->description;
    auto params = (hw_dpu_rank_allocation_parameters_t)(desc->_internals.data);
    uint8_t *base_addr = params->ptr_region;
    ReceiveFromRankMRAM((uint8_t**)&matrix->ptr[0], matrix->offset, base_addr, matrix->size);
    return DPU_OK;
  }

};

}

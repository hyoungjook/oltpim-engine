#pragma once

#include <immintrin.h>
#include <x86intrin.h>
#include <cinttypes>
#include <assert.h>
#include <libudev.h>
#include <iostream>

#ifndef OLTPIM_DISABLE_DIRECT_API
#define UPMEM_USE_DIRECT_MUX
#define UPMEM_USE_DIRECT_LAUNCH
#endif

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
#include "ufi_ci.h"
#include "ufi_ci_commands.h"
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
typedef struct _hw_dpu_rank_context_t {
    /* Hybrid mode: Address of control interfaces when memory mapped
     * Perf mode:   Base region address, mappings deal with offset to target control interfaces
     * Safe mode:   Buffer handed to the driver
     */
    uint64_t *control_interfaces;
} * hw_dpu_rank_context_t;
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

#define UPMEM_DIRECT_ASSERT(expr) (likely((expr) == (u32)DPU_OK) ? (void)0 : abort())

namespace upmem {
namespace internal {
namespace xeon {
/**
 * The optimized xeon_sp API modified from
 * * https://github.com/Loremkang/upmem-sdk-light/blob/7ca4de5b75fc0781321c8557815ffe3e247ffc18/src/pim_interface/direct_interface.hpp
 */
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

static void write_to_cis(
    dpu_region_address_translation *tr,
    void *base_region_addr, u64 *block_data) {
  u64 *ci_address = (u64*)((u8*)base_region_addr + 0x20000);
  byte_interleave_avx512(block_data, ci_address, true);
  tr->one_read = false;
}

static void read_from_cis(
    dpu_region_address_translation *tr,
    void *base_region_addr, u64 *block_data) {
  //#define NB_READS 3
  //const u8 nb_reads = tr->one_read ? NB_READS - 1 : NB_READS;
  // NOTE (hyoungjk): This significantly improves low-batch throughput
  // and LLC misses. It works well on the latest (v1B on upmemcloud9) UPMEM PIMs.
  constexpr u8 nb_reads = 1;
  u64 input[8];
  u64 *ci_address = (u64*)((u8*)base_region_addr + 0x20000 + 32 * 1024);
  for (u8 i = 0; i < nb_reads; ++i) {
    __builtin_ia32_clflushopt((uint8_t *)ci_address);
    __builtin_ia32_mfence();
    ((volatile u64*)input)[0] = *(ci_address + 0);
    ((volatile u64*)input)[1] = *(ci_address + 1);
    ((volatile u64*)input)[2] = *(ci_address + 2);
    ((volatile u64*)input)[3] = *(ci_address + 3);
    ((volatile u64*)input)[4] = *(ci_address + 4);
    ((volatile u64*)input)[5] = *(ci_address + 5);
    ((volatile u64*)input)[6] = *(ci_address + 6);
    ((volatile u64*)input)[7] = *(ci_address + 7);
  }
  byte_interleave_avx512(input, block_data, false);
  tr->one_read = true;
}

} // namespace xeon

namespace ufi {
/**
 * The optimized ufi API, modified from ufi.c and ufi_ci.c
 */
namespace ci {
static inline void commit_commands(dpu_rank_t *rank, u64 *commands) {
  xeon::write_to_cis(
    &((hw_dpu_rank_allocation_parameters_t)rank->description->_internals.data)->translate,
    ((hw_dpu_rank_context_t)rank->_internals)->control_interfaces,
    commands
  );
}

static inline void update_commands(dpu_rank_t *rank, u64 *commands) {
  xeon::read_from_cis(
    &((hw_dpu_rank_allocation_parameters_t)rank->description->_internals.data)->translate,
    ((hw_dpu_rank_context_t)rank->_internals)->control_interfaces,
    commands
  );
}

static inline void compute_masks(dpu_rank_t *rank, const u64 *commands,
		u64 *masks, u64 *expected, u8 *cis, bool *is_done) {
	u8 ci_mask = 0;
	u8 nr_cis = rank->description->hw.topology.nr_of_control_interfaces;
	for (u8 each_ci = 0; each_ci < nr_cis; ++each_ci) {
		if (commands[each_ci] != CI_EMPTY) {
			ci_mask |= (1 << each_ci);
			masks[each_ci] |= 0xFF0000FF00000000l;
			expected[each_ci] |= 0x000000FF00000000l;
		} else
			is_done[each_ci] = true;
	}
	*cis = ci_mask;
}

static inline bool determine_if_commands_are_finished(dpu_rank_t *rank,
    const u64 *data, const u64 *expected, const u64 *result_masks,
    u8 expected_color, bool *is_done) {
	dpu_control_interface_context *context = &rank->runtime.control_interface;
	u8 nr_cis = rank->description->hw.topology.nr_of_control_interfaces;
	u8 each_ci;
	u8 color, nb_bits_set, ci_mask, ci_color;

	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		if (!is_done[each_ci]) {
			u64 result = data[each_ci];

			/* The second case can happen when the debugger has restored the result */
			if ((result & result_masks[each_ci]) !=
				    expected[each_ci] &&
			    (result & CI_NOP) != CI_NOP) {
				return false;
			}

			/* From here, the result seems ok, we just need to check that the color is ok. */
			color = ((result & 0x00FF000000000000ULL) >> 48) & 0xFF;
			nb_bits_set = __builtin_popcount(color);
			ci_mask = 1 << each_ci;
			ci_color = expected_color & ci_mask;

			if (ci_color != 0) {
				if (nb_bits_set <= 3) {
					return false;
				}
			} else {
				if (nb_bits_set >= 5) {
					return false;
				}
			}

			is_done[each_ci] = true;

			/* We are in fault path, store this information */
			if (ci_color != 0) {
				nb_bits_set = 8 - nb_bits_set;
			}

			switch (nb_bits_set) {
			case 0:
				break;
			case 1:
				context->fault_decode |= ci_mask;
				break;
			case 2:
				context->fault_collide |= ci_mask;
				break;
			case 3:
				context->fault_collide |= ci_mask;
				context->fault_decode |= ci_mask;
				break;
			default:
				return false;
			}
		}
	}

	return true;
}

static void exec_cmd(struct dpu_rank_t *rank, u64 *commands)
{
	u8 expected_color;
	u64 result_masks[DPU_MAX_NR_CIS] = { 0 };
	u64 expected[DPU_MAX_NR_CIS] = { 0 };
	u64 *data = rank->data;
	bool is_done[DPU_MAX_NR_CIS] = { 0 };
	u8 ci_mask;
	bool in_progress, timeout;
	u32 nr_retries;
	u32 nr_retry_commit_cmd = 2;

	compute_masks(rank, commands, result_masks, expected, &ci_mask, is_done);
	expected_color = rank->runtime.control_interface.color & ci_mask;
  rank->runtime.control_interface.color ^= ci_mask;

	// TMP workaround: sometimes we have a CI timeout so we just need to resend the commands (see Jira SW-309)
	// The commands are well sent to the CI but sometimes acknowledgment seems to be not done
	// We can expect to have the issue by loading program to IRAM in loop for an hour
send_cmd:
	nr_retries = 100;
  ci::commit_commands(rank, commands);
	do {
    ci::update_commands(rank, data);
		in_progress = !determine_if_commands_are_finished(
			rank, data, expected, result_masks, expected_color,
			is_done);

		timeout = (nr_retries--) == 0;

		if (timeout && nr_retry_commit_cmd > 0) {
			nr_retry_commit_cmd--;
			goto send_cmd;
		}
	} while (in_progress && !timeout);
	if (in_progress) {
    UPMEM_DIRECT_ASSERT(DPU_ERR_TIMEOUT);
	}
	if (1) {
		/* All results are ready here, and still present when reading the control interfaces.
		 * We make sure that we have the correct results by reading again (we may have timing issues).
		 * todo(#85): this can be somewhat costly. We should try to integrate this additional read in a lower layer.
		 */
    ci::update_commands(rank, data);
	}
}

static inline void prepare_mask(u64 *buffer, u8 mask, u64 data) {
  for (u8 each_ci = 0; each_ci < DPU_MAX_NR_CIS; ++each_ci) {
    if (CI_MASK_ON(mask, each_ci)) {
			buffer[each_ci] = data;
		} else {
			buffer[each_ci] = CI_EMPTY;
		}
  }
}

static inline void exec_void_cmd(dpu_rank_t *rank, u64 *commands) {
  ufi::ci::exec_cmd(rank, commands);
}

static inline void exec_8bit_cmd(dpu_rank_t *rank, u64 *commands, u8 *results) {
  ufi::ci::exec_cmd(rank, commands);
  const u8 nr_cis = rank->description->hw.topology.nr_of_control_interfaces;
  for (u8 each_ci = 0; each_ci < nr_cis; ++each_ci) {
    if (commands[each_ci] != CI_EMPTY) {
      results[each_ci] = rank->data[each_ci];
    }
  }
}
} // namespace ci

static inline void exec_write_structure(dpu_rank_t *rank, u8 ci_mask, u64 structure) {
  const u8 nr_cis = rank->description->hw.topology.nr_of_control_interfaces;
  bool do_write_structure = false;
  for (u8 each_ci = 0; each_ci < nr_cis; ++each_ci) {
    if (rank->runtime.control_interface.slice_info[each_ci].structure_value != structure) {
      rank->runtime.control_interface.slice_info[each_ci].structure_value = structure;
      do_write_structure = true;
    }
  }
  if (do_write_structure) {
    u64 *cmds = rank->cmds;
    ufi::ci::prepare_mask(cmds, ci_mask, structure);
    ufi::ci::exec_void_cmd(rank, cmds);
  }
}

static inline void exec_void_frame(dpu_rank_t *rank, u8 ci_mask, u64 structure, u64 frame) {
  u64 *cmds = rank->cmds;
  ufi::exec_write_structure(rank, ci_mask, structure);
  ufi::ci::prepare_mask(cmds, ci_mask, frame);
  ufi::ci::exec_void_cmd(rank, cmds);
}

static inline void exec_8bit_frame(dpu_rank_t *rank, u8 ci_mask, u64 structure, u64 frame, u8 *results) {
  u64 *cmds = rank->cmds;
  ufi::exec_write_structure(rank, ci_mask, structure);
  ufi::ci::prepare_mask(cmds, ci_mask, frame);
  ufi::ci::exec_8bit_cmd(rank, cmds, results);
}

static inline u64 ci_dma_ctrl_write_frame(u8 address, u8 data)
{
	u8 b0 = 0x60 | (((address) >> 4) & 0x0F);
	u8 b1 = 0x60 | (((address) >> 0) & 0x0F);
	u8 b2 = 0x60 | (((data) >> 4) & 0x0F);
	u8 b3 = 0x60 | (((data) >> 0) & 0x0F);
	u8 b4 = 0x60;
	u8 b5 = 0x20;

	return CI_DMA_CTRL_WRITE_FRAME(b0, b1, b2, b3, b4, b5);
}

static inline void write_dma_ctrl(dpu_rank_t *rank, u8 ci_mask, u8 address, u8 data) {
  ufi::exec_void_frame(
  rank, ci_mask, CI_DMA_CTRL_WRITE_STRUCT,
  ci_dma_ctrl_write_frame(address, data));
}

static inline void write_dma_ctrl_datas(dpu_rank_t *rank, u8 ci_mask, u8 address, u8 *datas) {
  const u8 nr_cis = rank->description->hw.topology.nr_of_control_interfaces;
  ufi::exec_write_structure(rank, ci_mask, CI_DMA_CTRL_WRITE_STRUCT);
  u64 *frames = rank->cmds;
  for (u8 each_ci = 0; each_ci < nr_cis; ++each_ci) {
    frames[each_ci] = ci_dma_ctrl_write_frame(address, datas[each_ci]);
  }
  ufi::ci::exec_void_cmd(rank, frames);
}

static inline void read_dma_ctrl(dpu_rank_t *rank, u8 ci_mask, u8 *data) {
  ufi::exec_8bit_frame(
  rank, ci_mask, CI_DMA_CTRL_READ_STRUCT,
  CI_DMA_CTRL_READ_FRAME, data);
}

static inline void clear_dma_ctrl(dpu_rank_t *rank, u8 ci_mask) {
  ufi::exec_void_frame(
    rank, ci_mask, CI_DMA_CTRL_CLEAR_STRUCT,
    CI_DMA_CTRL_CLEAR_FRAME);
}

static inline void thread_boot(dpu_rank_t *rank, u8 ci_mask, u8 thread) {
  // assume prev == NULL
  ufi::exec_void_frame(
    rank, ci_mask, CI_THREAD_BOOT_STRUCT,
    CI_THREAD_BOOT_FRAME(thread));
}

static inline void read_dpu_run(dpu_rank_t *rank, u8 ci_mask, u8 *run) {
  ufi::exec_8bit_frame(
    rank, ci_mask, CI_DPU_RUN_STATE_READ_STRUCT,
    CI_DPU_RUN_STATE_READ_FRAME, run);
}

static inline void read_dpu_fault(dpu_rank_t *rank, u8 ci_mask, u8 *fault) {
  ufi::exec_8bit_frame(
    rank, ci_mask, CI_DPU_FAULT_STATE_READ_STRUCT,
    CI_DPU_FAULT_STATE_READ_FRAME, fault);
}
} // namespace ufi
} // namespace internal

namespace direct {
/**
 * Direct interface
 */
namespace mux {
/**
 * The optimized mux_switch API.
 * The main contribution is that we removed unnecessary call to
 * ufi_select_dpu_even_disabled() inside dpu_check_wavegen_mux_status_for_rank().
 * !!!!! ASSUMES ALL DPUS ARE ENABLED !!!!!
 * If not, it may fail. (not tested)
 */
[[maybe_unused]] static bool is_switch_required(dpu_rank_t *rank, bool mux_for_host) {
  dpu_description_t desc = rank->description;
  const uint8_t nr_cis = desc->hw.topology.nr_of_control_interfaces;
  const uint8_t nr_dpus_per_ci = desc->hw.topology.nr_of_dpus_per_control_interface;
  for (uint8_t each_ci = 0; each_ci < nr_cis; ++each_ci) {
    dpu_bitfield_t host_mux_mram_state =
      rank->runtime.control_interface.slice_info[each_ci].host_mux_mram_state;
    if ((mux_for_host && __builtin_popcount(host_mux_mram_state) < nr_dpus_per_ci) ||
        (!mux_for_host && host_mux_mram_state)) {
      return true;
    }
  }
  return false;
}

[[maybe_unused]] static void switch_begin(dpu_rank_t *rank, bool mux_for_host) {
  dpu_description_t desc = rank->description;
  const uint8_t nr_cis = desc->hw.topology.nr_of_control_interfaces;
  const uint8_t nr_dpus_per_ci = desc->hw.topology.nr_of_dpus_per_control_interface;
  for (uint8_t each_ci = 0; each_ci < nr_cis; ++each_ci) {
    rank->runtime.control_interface.slice_info[each_ci].host_mux_mram_state =
      (mux_for_host ? ((1 << nr_dpus_per_ci) - 1) : 0x0);
  }
  uint8_t ci_mask = ALL_CIS;
  UPMEM_DIRECT_ASSERT(ufi_select_all_even_disabled(rank, &ci_mask));
  assert(ci_mask == ALL_CIS);
  // ufi_set_mram_mux
  u8 mux_value[DPU_MAX_NR_CIS];
  for (u8 each_ci = 0; each_ci < nr_cis; ++each_ci) {
    mux_value[each_ci] = mux_for_host ? 0 : 1;
  }
  internal::ufi::write_dma_ctrl_datas(rank, ci_mask, 0x80, mux_value);
  internal::ufi::write_dma_ctrl(rank, ci_mask, 0x81, 0);
  internal::ufi::write_dma_ctrl_datas(rank, ci_mask, 0x82, mux_value);
  internal::ufi::write_dma_ctrl_datas(rank, ci_mask, 0x84, mux_value);
  internal::ufi::clear_dma_ctrl(rank, ci_mask);
}

[[maybe_unused]] static void switch_sync(dpu_rank_t *rank, bool mux_for_host) {
  dpu_description_t desc = rank->description;
  const uint8_t nr_cis = desc->hw.topology.nr_of_control_interfaces;
  const uint8_t nr_dpus_per_ci = desc->hw.topology.nr_of_dpus_per_control_interface;
  uint8_t ci_mask = ALL_CIS;
  // dpu_check_wavegen_mux_status_for_rank
  internal::ufi::write_dma_ctrl(rank, ci_mask, 0xFF, 0x02);
  internal::ufi::clear_dma_ctrl(rank, ci_mask);
  uint8_t result_array[DPU_MAX_NR_CIS];
  const uint8_t wavegen_expected = mux_for_host ? 0x00 : ((1 << 0) | (1 << 1));
  for (uint8_t each_dpu = 0; each_dpu < nr_dpus_per_ci; ++each_dpu) {
    uint32_t timeout = 100;
    bool should_retry = false;
    do {
      internal::ufi::read_dma_ctrl(rank, ci_mask, result_array);
      for (uint8_t each_ci = 0; each_ci < nr_cis; ++each_ci) {
        if ((result_array[each_ci] & 0x7B) != wavegen_expected) {
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

[[maybe_unused]] static void switch_rank(dpu_rank_t *rank, bool mux_for_host) {
#if defined(UPMEM_USE_DIRECT_MUX)
  if (!is_switch_required(rank, mux_for_host)) return;
  switch_begin(rank, mux_for_host);
  switch_sync(rank, mux_for_host);
#else
  DPU_ASSERT(dpu_switch_mux_for_rank(rank, mux_for_host));
#endif
}
} // namespace mux

namespace launch {
/**
 * The direct-inlined launch API.
 */
[[maybe_unused]] static void boot(dpu_rank_t *rank) {
  mux::switch_rank(rank, false);
#if defined(UPMEM_USE_DIRECT_LAUNCH)
  uint8_t ci_mask = ALL_CIS;
  UPMEM_DIRECT_ASSERT(ufi_select_all(rank, &ci_mask));
  internal::ufi::thread_boot(rank, ci_mask, DPU_BOOT_THREAD);
#else
  DPU_ASSERT(dpu_boot_rank(rank));
#endif
}

[[maybe_unused]] static void poll_status(dpu_rank_t *rank, bool *done, bool *fault) {
#if defined(UPMEM_USE_DIRECT_LAUNCH)
  uint8_t ci_mask = ALL_CIS;
  dpu_bitfield_t poll_running[DPU_MAX_NR_CIS];
  UPMEM_DIRECT_ASSERT(ufi_select_all(rank, &ci_mask));
  internal::ufi::read_dpu_run(rank, ci_mask, poll_running);
#ifndef NDEBUG
  dpu_bitfield_t poll_fault[DPU_MAX_NR_CIS];
  internal::ufi::read_dpu_fault(rank, ci_mask, poll_fault);
#endif
  const uint8_t nr_cis = rank->description->hw.topology.nr_of_control_interfaces;
  *done = true;
  *fault = false;
  for (uint8_t each_ci = 0; each_ci < nr_cis; ++each_ci) {
    dpu_selected_mask_t mask_all = rank->runtime.control_interface.slice_info[each_ci].enabled_dpus;
#ifndef NDEBUG
    dpu_bitfield_t ci_fault = poll_fault[each_ci] & mask_all;
#else
    const dpu_bitfield_t ci_fault = 0;
#endif
    dpu_bitfield_t ci_running = (poll_running[each_ci] & mask_all) & (~ci_fault);
    *done = *done && (ci_running == 0);
    *fault = *fault || (ci_fault != 0);
  }
#else
  DPU_ASSERT(dpu_poll_rank(rank));
  DPU_ASSERT(dpu_status_rank(rank, done, fault));
#endif
}
} // namespace launch

namespace copy {
/**
 * The direct PIM-CPU copy API using internal::xeon.
 * It allows CPU-PIM copy to occur entirely within the calling thread
 */
[[maybe_unused]] static dpu_error_t copy_to_mrams(dpu_rank_t *rank, dpu_transfer_matrix *matrix) {
  mux::switch_rank(rank, true);
  dpu_description_t desc = rank->description;
  auto params = (hw_dpu_rank_allocation_parameters_t)(desc->_internals.data);
  uint8_t *base_addr = params->ptr_region;
  internal::xeon::SendToRankMRAM((uint8_t**)&matrix->ptr[0], matrix->offset, base_addr, matrix->size);
  return DPU_OK;
}

[[maybe_unused]] static dpu_error_t copy_from_mrams(dpu_rank_t *rank, dpu_transfer_matrix *matrix) {
  mux::switch_rank(rank, true);
  dpu_description_t desc = rank->description;
  auto params = (hw_dpu_rank_allocation_parameters_t)(desc->_internals.data);
  uint8_t *base_addr = params->ptr_region;
  internal::xeon::ReceiveFromRankMRAM((uint8_t**)&matrix->ptr[0], matrix->offset, base_addr, matrix->size);
  return DPU_OK;
}
} // namespace copy

} // namespace direct

}

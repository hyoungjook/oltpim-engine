#include "upmem_rank.hpp"
#include "upmem_direct.hpp"

extern "C" {
  #include <dpu.h>
  #include <dpu_config.h>
  #include <dpu_debug.h>
  #include <dpu_elf.h>
  #include <dpu_loader.h>
  #include <dpu_log.h>
  #include <dpu_management.h>
  #include <dpu_memory.h>
  #include <dpu_profiler.h>
  #include <dpu_program.h>
  #include <dpu_runner.h>
  #include <dpu_target_macros.h>
}

namespace upmem {

std::mutex rank::_alloc_mutex;

void rank::alloc(const char *profile) {
  _rank = (dpu_rank_t*)static_alloc(profile);
  DPU_ASSERT(_rank ? DPU_OK : DPU_ERR_ALLOCATION);
  init(_rank);
}

void *rank::static_alloc(const char *profile) {
  std::unique_lock lk(_alloc_mutex);
  dpu_rank_t *rank;
  dpu_error_t status = dpu_get_rank_of_type(profile, &rank);
  if (status != DPU_OK) {
    if (status == DPU_ERR_ALLOCATION) rank = nullptr;
    else DPU_ASSERT(status);
  }
  if (rank) {
    DPU_ASSERT(dpu_reset_rank(rank));
  }
  return (void*)rank;
}

void rank::init(void *dpu_rank) {
  _rank = (dpu_rank_t*)dpu_rank;

  // Count dpus
  struct dpu_t *dpu;
  int num_dpus = 0;
  STRUCT_DPU_FOREACH(_rank, dpu) {
    if (dpu_is_enabled(dpu)) {
      ++num_dpus;
    }
  }
  _num_dpus = num_dpus;

  // Save information
  _numa_node = numa_node_of(_rank);
  dpu_description_t dpu_desc = dpu_get_description(_rank);
  _mram_size = dpu_desc->hw.memories.mram_size;
}

rank::~rank() {
  for (dpu_transfer_matrix *m: _registered_transfers) {
    free(m);
  }
  _registered_transfers.clear();
  if (_program) dpu_free_program(_program);
  if (_rank) DPU_ASSERT(dpu_free_rank(_rank));
}

void rank::load(const char *binary_path) {
  struct dpu_t *dpu;
  _program = (struct dpu_program_t*)malloc(sizeof(*_program));
  _binary_path = binary_path;
  dpu_elf_file_t elf_info;
  struct _dpu_loader_context_t loader_context;

  dpu_init_program_ref(_program);
  dpu_take_program_ref(_program);
  DPU_ASSERT(dpu_load_elf_program(&elf_info, binary_path, _program, _mram_size));
  STRUCT_DPU_FOREACH(_rank, dpu) {
    dpu_take_program_ref(_program);
    dpu_set_program(dpu, _program);
  }
  DPU_ASSERT(dpu_fill_profiling_info(_rank,
    (iram_addr_t)_program->mcount_address,
    (iram_addr_t)_program->ret_mcount_address,
    (wram_addr_t)_program->thread_profiling_address,
    (wram_addr_t)_program->perfcounter_end_value_address,
    _program->profiling_symbols));
  dpu_loader_fill_rank_context(&loader_context, _rank);
  DPU_ASSERT(dpu_elf_load(elf_info, &loader_context));

  dpu_elf_close(elf_info);
}

bool rank::launch(bool async) {
  direct::launch::boot(_rank);
  if (!async) {
    bool fault = false;
    while (true) {
      if (is_done(&fault)) break;
    }
    return !fault;
  }
  return true;
}

bool rank::is_done(bool *fault) {
  bool _done, _fault;
  direct::launch::poll_status(_rank, &_done, &_fault);
  if (fault) *fault = _fault;
  return _done;
}

uint32_t rank::register_dpu_transfer(const char *symbol, void **buffers, bool broadcast) {
  uint32_t address = (uint32_t)-1;
  {
    // Find symbol address
    if (!_program) DPU_ASSERT(DPU_ERR_NO_PROGRAM_LOADED);
    uint32_t nr_symbols = _program->symbols->nr_symbols;
    for (uint32_t each_symbol = 0; each_symbol < nr_symbols; ++each_symbol) {
      dpu_elf_symbol_t *s = _program->symbols->map + each_symbol;
      if (strcmp(s->name, symbol) == 0) {
        address = s->value;
        break;
      }
    }
    DPU_ASSERT((address != (uint32_t)-1) ? DPU_OK : DPU_ERR_UNKNOWN_SYMBOL);
  }
  // Register transfer
  uint32_t transfer_id = _registered_transfers.size();
  auto *matrix = (dpu_transfer_matrix*)malloc(sizeof(dpu_transfer_matrix));
  _registered_transfers.push_back(matrix);
  // address: store the original address for now. It will be modified to
  // aligned & offsetted address later.
  matrix->offset = address;
  matrix->type = DPU_DEFAULT_XFER_MATRIX;
  // buffers
  {
    struct dpu_t *dpu;
    int each_dpu = 0, each_enabled_dpu = 0;
    STRUCT_DPU_FOREACH(_rank, dpu) {
      if (dpu_is_enabled(dpu)) {
        // If broadcast, treat void** buffers as just single void* pointer.
        // Otherwise, void** buffers is the array of void* pointers.
        matrix->ptr[each_dpu] = (broadcast ? (void*)buffers : buffers[each_enabled_dpu]);
        ++each_enabled_dpu;
      }
      else {
        matrix->ptr[each_dpu] = nullptr;
      }
      ++each_dpu;
    }
    DPU_ASSERT(each_enabled_dpu == _num_dpus ? DPU_OK : DPU_ERR_INTERNAL);
  }
  return transfer_id;
}

void rank::copy(uint32_t transfer_id, uint32_t length, bool direction_to_dpu, uint32_t symbol_offset) {
  DPU_ASSERT(transfer_id < _registered_transfers.size() ? DPU_OK : DPU_ERR_INVALID_SYMBOL_ACCESS);
  auto *matrix = _registered_transfers[transfer_id];

  // Backup offset
  const uint32_t original_address = matrix->offset;

  // Adjust offset & Select copy function
  memory_type mem_type = fill_transfer_matrix(matrix, symbol_offset, length);
  const auto copy_fn = (dpu_error_t (*)(dpu_rank_t*,dpu_transfer_matrix*))
    get_copy_fn(mem_type, direction_to_dpu);

  // Do copy
  DPU_ASSERT(copy_fn(_rank, matrix));

  // Restore offset
  matrix->offset = original_address;
}

static std::mutex log_print_mutex;

void rank::log_read(FILE *stream, bool fault_only, int dpu_id) {
  std::unique_lock lck(log_print_mutex);
  struct dpu_t *dpu;
  int each_dpu = -1;
  STRUCT_DPU_FOREACH(_rank, dpu) {
    ++each_dpu;
    if (fault_only) {
      bool done = false, fault = false;
      DPU_ASSERT(dpu_status_dpu(dpu, &done, &fault));
      if (!fault) continue;
    }
    if (dpu_id >= 0) {
      if (each_dpu != dpu_id) continue;
    }
    fprintf(stream, "===\n");
    DPU_ASSERT(dpulog_read_for_dpu(dpu, stream));
  }
}

rank::memory_type rank::fill_transfer_matrix(dpu_transfer_matrix *matrix,
    uint32_t symbol_offset, uint32_t length) {
  // Symbol type
  struct _mask_align {dpu_mem_max_addr_t mask, align; memory_type type;};
  constexpr _mask_align IRAM = {0x80000000u, 3u, memory_type::IRAM};
  constexpr _mask_align MRAM = {0x08000000u, 0u, memory_type::MRAM};
  constexpr _mask_align WRAM = {0x00000000u, 2u, memory_type::WRAM};

  dpu_mem_max_addr_t address = matrix->offset;
  _mask_align mem_type =
    ((address & IRAM.mask) == IRAM.mask) ? IRAM :
    (((address & MRAM.mask) == MRAM.mask) ? MRAM : WRAM);
  address = ((address + symbol_offset) & (~mem_type.mask)) >> mem_type.align;
  length = length >> mem_type.align;

  matrix->offset = address;
  matrix->size = length;
  matrix->type = DPU_DEFAULT_XFER_MATRIX;

  return mem_type.type;
}

void *rank::get_copy_fn(memory_type mem_type, bool direction_to_dpu) {
  switch (mem_type) {
  case memory_type::MRAM: {
    if (direction_to_dpu)
      //return (void*)dpu_copy_to_mrams;              // MRAM, CPU_TO_PIM
      return (void*)upmem::direct::copy::copy_to_mrams;
    else
      //return (void*)dpu_copy_from_mrams;            // MRAM, PIM_TO_CPU
      return (void*)upmem::direct::copy::copy_from_mrams;
  }
  case memory_type::WRAM: {
    if (direction_to_dpu)
      return (void*)dpu_copy_to_wram_for_matrix;    // WRAM, CPU_TO_PIM
    else
      return (void*)dpu_copy_from_wram_for_matrix;  // WRAM, PIM_TO_CPU
  }
  case memory_type::IRAM: {
    if (direction_to_dpu)
      return (void*)dpu_copy_to_iram_for_matrix;    // IRAM, CPU_TO_PIM
    else
      return (void*)dpu_copy_from_iram_for_matrix;  // IRAM, PIM_TO_CPU
  }
  }
  return nullptr;
}

int rank::numa_node_of(void *dpu_rank) {
  int numa_node = dpu_get_rank_numa_node((dpu_rank_t*)dpu_rank);
  if (numa_node == -1) numa_node = 0; // if the backend is functional simulator
  return numa_node;
}

void rank::static_free(void *dpu_rank) {
  DPU_ASSERT(dpu_free_rank((dpu_rank_t*)dpu_rank));
}

void rank::handle_fault() {
  struct dpu_t *dpu = nullptr;
  bool running = false, fault = false;
  STRUCT_DPU_FOREACH(_rank, dpu) {
    DPU_ASSERT(dpu_poll_dpu(dpu, &running, &fault));
    if (fault) {
      break;
    }
  }
  if (dpu) {
    // dpu points to the first faulty dpu
    dpu_id_t rank_id = dpu_get_rank_id(_rank) & DPU_TARGET_MASK;
    dpu_slice_id_t slice_id = dpu_get_slice_id(dpu);
    dpu_member_id_t member_id = dpu_get_member_id(dpu);
    printf("One (or more) DPUs in fault. To attach to the first faulty DPU, run:\n");
    printf("dpu-lldb-attach-dpu %u.%u.%u %s\n",
      rank_id, slice_id, member_id, _binary_path);
  }
}

}

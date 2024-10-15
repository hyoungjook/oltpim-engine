#include "upmem_rank.hpp"

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
  if (_program) dpu_free_program(_program);
  if (_rank) DPU_ASSERT(dpu_free_rank(_rank));
}

void rank::load(const char *binary_path) {
  struct dpu_t *dpu;
  _program = (struct dpu_program_t*)malloc(sizeof(*_program));
  dpu_elf_file_t elf_info;
  struct _dpu_loader_context_t loader_context;

  dpu_init_program_ref(_program);
  dpu_take_program_ref(_program);
  DPU_ASSERT(dpu_load_elf_program(&elf_info, binary_path, _program, _mram_size));
  STRUCT_DPU_FOREACH(_rank, dpu) {
    dpu_take_program_ref(_program);
    dpu_set_program(dpu, _program);
  }
  dpu_loader_fill_rank_context(&loader_context, _rank);
  DPU_ASSERT(dpu_elf_load(elf_info, &loader_context));

  dpu_elf_close(elf_info);
}

bool rank::launch(bool async) {
  DPU_ASSERT(dpu_boot_rank(_rank));
  if (!async) {
    bool done = false, fault = false;
    while (true) {
      done = is_done(&fault);
      if (fault) return false;
      if (done) return true;
    }
  }
  return true;
}

bool rank::is_done(bool *fault) {
  bool _done, _fault;
  DPU_ASSERT(dpu_poll_rank(_rank));
  DPU_ASSERT(dpu_status_rank(_rank, &_done, &_fault));
  if (fault) *fault = _fault;
  return _done || _fault;
}

uint32_t rank::register_dpu_symbol(const char *symbol) {
  if (!_program) DPU_ASSERT(DPU_ERR_NO_PROGRAM_LOADED);
  uint32_t nr_symbols = _program->symbols->nr_symbols;
  for (uint32_t each_symbol = 0; each_symbol < nr_symbols; each_symbol++) {
    dpu_elf_symbol_t *s = _program->symbols->map + each_symbol;
    if (strcmp(s->name, symbol) == 0) {
      dpu_symbol_t found_symbol;
      found_symbol.address = s->value;
      found_symbol.size = s->size;
      uint32_t symbol_id = _registered_symbols.size();
      _registered_symbols.push_back(found_symbol);
      return symbol_id;
    }
  }
  DPU_ASSERT(DPU_ERR_UNKNOWN_SYMBOL);
  return -1;
}

void rank::copy(uint32_t symbol_id, void **buffers, uint32_t length,
    bool direction_to_dpu, uint32_t symbol_offset) {
  if (symbol_id >= _registered_symbols.size())
    DPU_ASSERT(DPU_ERR_INVALID_SYMBOL_ACCESS);

  dpu_transfer_matrix matrix;
  auto mem_type = fill_transfer_matrix(&matrix, symbol_id, symbol_offset, length);
  using copy_fn_type = dpu_error_t (*)(dpu_rank_t*, dpu_transfer_matrix*);
  const auto copy_fn = (copy_fn_type)get_copy_fn(mem_type, direction_to_dpu);

  // Fill ptr
  {
    struct dpu_t *dpu;
    int each_dpu = 0, each_enabled_dpu = 0;
    STRUCT_DPU_FOREACH(_rank, dpu) {
      if (dpu_is_enabled(dpu)) {
        matrix.ptr[each_dpu] = buffers[each_enabled_dpu];
        ++each_enabled_dpu;
      }
      ++each_dpu;
    }
    DPU_ASSERT(each_enabled_dpu == _num_dpus ? DPU_OK : DPU_ERR_INTERNAL);
  }

  DPU_ASSERT(copy_fn(_rank, &matrix));
}

void rank::broadcast(uint32_t symbol_id, void *buffer, uint32_t length,
    uint32_t symbol_offset) {
  if (symbol_id >= _registered_symbols.size())
    DPU_ASSERT(DPU_ERR_INVALID_SYMBOL_ACCESS);

  dpu_transfer_matrix matrix;
  auto mem_type = fill_transfer_matrix(&matrix, symbol_id, symbol_offset, length);
  using copy_fn_type = dpu_error_t (*)(dpu_rank_t*, dpu_transfer_matrix*);
  const auto copy_fn = (copy_fn_type)get_copy_fn(mem_type, true);

  // Fill ptr
  {
    struct dpu_t *dpu;
    int each_dpu = 0;
    STRUCT_DPU_FOREACH(_rank, dpu) {
      if (dpu_is_enabled(dpu)) {
        matrix.ptr[each_dpu] = buffer;
      }
      ++each_dpu;
    }
  }

  DPU_ASSERT(copy_fn(_rank, &matrix));
}

void rank::log_read(FILE *stream) {
  struct dpu_t *dpu;
  STRUCT_DPU_FOREACH(_rank, dpu) {
    DPU_ASSERT(dpulog_read_for_dpu(dpu, stream));
  }
}

rank::memory_type rank::fill_transfer_matrix(dpu_transfer_matrix *matrix,
    uint32_t symbol_id, uint32_t symbol_offset, uint32_t length) {
  memset(matrix, 0, sizeof(*matrix));

  // Symbol type
  struct _mask_align {dpu_mem_max_addr_t mask, align; memory_type type;};
  constexpr _mask_align IRAM = {0x80000000u, 3u, memory_type::IRAM};
  constexpr _mask_align MRAM = {0x08000000u, 0u, memory_type::MRAM};
  constexpr _mask_align WRAM = {0x00000000u, 2u, memory_type::WRAM};

  dpu_mem_max_addr_t address = _registered_symbols[symbol_id].address;
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
      return (void*)dpu_copy_to_mrams;              // MRAM, CPU_TO_PIM
    else
      return (void*)dpu_copy_from_mrams;            // MRAM, PIM_TO_CPU
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

}

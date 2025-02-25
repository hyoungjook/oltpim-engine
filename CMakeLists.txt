cmake_minimum_required(VERSION 3.20)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(oltpim-engine)
include_directories(include include/dpu include/host)

set(COMMON_CFLAGS -Wall -Werror)

# DPU codes: only *.c
set(DPU_BINARY oltpim_dpu)
### Larger NR_TASKLETS accelerates PIM program but increases mux switch latency
set(DPU_CFLAGS ${COMMON_CFLAGS} -DNR_TASKLETS=4)
#set(DPU_CFLAGS ${COMMON_CFLAGS} -DNR_TASKLETS=2 -pg) # if using dpu-profiling
set(DPU_CFLAGS_DEBUG -O3 -g3) # -O3 required to fit in IRAM
set(DPU_CFLAGS_RELEASE -O3 -DNDEBUG)
set(DPU_C_COMPILER dpu-upmem-dpurte-clang)
file(GLOB DPU_FILES dpu/*.c)

get_property(INCLUDE_DIRS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
set(INCLUDE_FLAGS "")
foreach(dir ${INCLUDE_DIRS})
    set(INCLUDE_FLAGS ${INCLUDE_FLAGS} "-I${dir}")
endforeach()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(DPU_CFLAGS ${DPU_CFLAGS} ${DPU_CFLAGS_RELEASE})
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(DPU_CFLAGS ${DPU_CFLAGS} ${DPU_CFLAGS_DEBUG})
endif()
add_custom_command(
  OUTPUT ${DPU_BINARY}
  COMMAND ${DPU_C_COMPILER} ${INCLUDE_FLAGS} ${DPU_CFLAGS} -o ${DPU_BINARY} ${DPU_FILES}
  DEPENDS ${DPU_FILES}
)
add_custom_target(build_dpu_binary ALL DEPENDS ${DPU_BINARY})
file(REAL_PATH ${DPU_BINARY} DPU_BINARY_PATH BASE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Host codes: only *.cpp
configure_file(test/analyze_sample_dpu.py analyze_sample_dpu.py COPYONLY)
file(REAL_PATH analyze_sample_dpu.py ANALYZE_SAMPLE_DPU_PATH BASE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

include_directories(
  upmem-sdk-runtime/commons/include
  upmem-sdk-runtime/commons/src/properties
  upmem-sdk-runtime/hw/src/commons
  upmem-sdk-runtime/hw/src/rank
  upmem-sdk-runtime/ufi/include/ufi
  upmem-sdk-runtime/verbose/src
)
execute_process(COMMAND dpu-pkg-config --cflags dpu
  OUTPUT_VARIABLE CFLAGS_DPU_CONFIG
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(COMMAND dpu-pkg-config --libs dpu
  OUTPUT_VARIABLE LDFLAGS_DPU_CONFIG
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CFLAGS
  ${COMMON_CFLAGS} ${CFLAGS_DPU_CONFIG}
  -DDPU_BINARY=\"${DPU_BINARY_PATH}\"
  -DANALYZE_SAMPLE_DPU_PATH=\"${ANALYZE_SAMPLE_DPU_PATH}\"
  -std=c++20 -fcoroutines -march=native
)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -g")
set(LDFLAGS ${LDFLAGS_DPU_CONFIG} pthread stdc++fs)
file(GLOB SRC_FILES src/*.cpp)
add_library(oltpim STATIC ${SRC_FILES})
add_dependencies(oltpim build_dpu_binary)
target_compile_options(oltpim PRIVATE ${CFLAGS})
target_link_libraries(oltpim PRIVATE ${LDFLAGS} numa)

add_library(oltpim_nodirect STATIC ${SRC_FILES})
add_dependencies(oltpim_nodirect build_dpu_binary)
target_compile_options(oltpim_nodirect PRIVATE ${CFLAGS} -DOLTPIM_DISABLE_DIRECT_API)
target_link_libraries(oltpim_nodirect PRIVATE ${LDFLAGS} numa)

# Test codes
add_executable(test_ycsb test/test_ycsb.cpp)
target_compile_options(test_ycsb PRIVATE ${CFLAGS})
target_link_libraries(test_ycsb PRIVATE oltpim)

add_executable(simulate_sample_dpu test/simulate_sample_dpu.cpp)
target_compile_options(simulate_sample_dpu PRIVATE ${COMMON_CFLAGS} ${CFLAGS_DPU_CONFIG})
target_link_libraries(simulate_sample_dpu PRIVATE ${LDFLAGS_DPU_CONFIG})

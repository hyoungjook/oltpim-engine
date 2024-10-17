cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(oltpim-engine)
include_directories(include include/dpu include/host)

set(COMMON_CFLAGS -Wall -Werror)

# DPU codes: only *.c
set(DPU_BINARY oltpim_dpu)
set(DPU_CFLAGS ${COMMON_CFLAGS} -DNR_TASKLETS=1)
set(CMAKE_C_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(DPU_LDFLAGS )
set(CMAKE_C_COMPILER dpu-upmem-dpurte-clang)
add_executable(${DPU_BINARY})
file(GLOB DPU_FILES dpu/*.c)
target_sources(${DPU_BINARY} PRIVATE ${DPU_FILES})
target_compile_options(${DPU_BINARY} PRIVATE ${DPU_CFLAGS})
target_link_libraries(${DPU_BINARY} PRIVATE ${DPU_LDFLAGS})

# Host codes: only *.cpp
set(HOST_BINARY oltpim)
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
  -DDPU_BINARY=\"${DPU_BINARY}\"
  -std=c++17
)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(LDFLAGS ${LDFLAGS_DPU_CONFIG} pthread stdc++fs)
add_executable(${HOST_BINARY})
file(GLOB SRC_FILES src/*.cpp)
target_sources(${HOST_BINARY} PRIVATE ${SRC_FILES})
target_compile_options(${HOST_BINARY} PRIVATE ${CFLAGS})

find_library(LIBNUMA numa REQUIRED)
target_link_libraries(${HOST_BINARY} PRIVATE ${LDFLAGS} ${LIBNUMA})



cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(oltpim-engine)
include_directories(include include/dpu include/host)

set(COMMON_CFLAGS -Wall -Werror)

# DPU codes: only *.c
set(DPU_BINARY oltpim_dpu)
set(DPU_CFLAGS ${COMMON_CFLAGS} -DNR_TASKLETS=8)
set(DPU_CFLAGS_DEBUG -O0 -g3)
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
  -std=c++20 -fcoroutines
)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -g3 -fno-omit-frame-pointer")
set(LDFLAGS ${LDFLAGS_DPU_CONFIG} pthread stdc++fs)
add_executable(${HOST_BINARY})
file(GLOB SRC_FILES src/*.cpp)
target_sources(${HOST_BINARY} PRIVATE ${SRC_FILES})
target_compile_options(${HOST_BINARY} PRIVATE ${CFLAGS})

find_library(LIBNUMA numa REQUIRED)
target_link_libraries(${HOST_BINARY} PRIVATE ${LDFLAGS} ${LIBNUMA})



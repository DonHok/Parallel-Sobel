cmake_minimum_required(VERSION 2.8)
project(PPP)

find_package(MPI)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -std=c11 -fopenmp -D_POSIX_C_SOURCE=200809L")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Adding utility options
option(PPP_DBG_MODE "Enable debug mode to ease debugging." off)
if (PPP_DBG_MODE)
  set(DBG_FLAGS "-O0 -g -fno-omit-frame-pointer")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${DBG_FLAGS}")
endif (PPP_DBG_MODE)
option(PPP_FAST_MODE "Enable compiler flags that speed up the binary." on)

set(FAST_FLAGS "-O3 -DNDEBUG -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FAST_FLAGS}")
if (PPP_FAST_MODE AND NOT PPP_DBG_MODE)
  set(FAST_FLAGS "-O3 -DNDEBUG -march=native")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FAST_FLAGS}")
endif (PPP_FAST_MODE AND NOT PPP_DBG_MODE)

include_directories(
  include
  ${MPI_C_INCLUDE_PATH}
  )

add_subdirectory(src)

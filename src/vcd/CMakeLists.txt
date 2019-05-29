set(PPP_MAIN_SRC
  main.c
  )

# Add user code source files here
set(PPP_CODE_SRC
  single.c
  parallel.c
  )

add_library(vcd_parts
  ${PPP_CODE_SRC}
  )

add_executable(vcd ${PPP_MAIN_SRC})
target_link_libraries(vcd
  vcd_parts
  ppp_pnm
  ${MPI_C_LIBRARIES}
  m
  )

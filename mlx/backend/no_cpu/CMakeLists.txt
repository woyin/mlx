target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common/load.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/../common/common.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/../common/compiled.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/../common/compiled_nocpu.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/../common/reduce_utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/../common/slicing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/../common/utils.cpp)

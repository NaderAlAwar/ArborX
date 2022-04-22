PYBIND11_ADD_MODULE(PyArborX ArborX_LinearBVH_pybind.cpp)
target_link_libraries(PyArborX PRIVATE ArborX::ArborX)
target_include_directories(PyArborX INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/details>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/details>
)

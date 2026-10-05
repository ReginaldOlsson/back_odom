cmake_minimum_required(VERSION 3.22)
project(back_odom)

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(Sophus REQUIRED)
find_package(TBB REQUIRED)
find_package(kiss_icp_cpp REQUIRED)
find_package(OpenMP REQUIRED)

option(BACK_ODOM_ENABLE_SIMD "Compile host code with native SIMD (AVX2/FMA via -march=native)" ON)
if(BACK_ODOM_ENABLE_SIMD)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|aarch64|arm64)$")
    set(BACK_ODOM_SIMD_FLAGS -march=native -mtune=native)
  else()
    set(BACK_ODOM_SIMD_FLAGS)
    message(WARNING "BACK_ODOM_ENABLE_SIMD is ON but no SIMD flags are known for ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  # Match kiss_icp_cpp's 32-byte Eigen ABI (required when both enable AVX).
  set(BACK_ODOM_SIMD_DEFINES EIGEN_MAX_ALIGN_BYTES=32 EIGEN_MAX_STATIC_ALIGN_BYTES=32)
  if(BACK_ODOM_SIMD_FLAGS)
    message(STATUS "back_odom SIMD enabled: ${BACK_ODOM_SIMD_FLAGS}; Eigen align=32")
  endif()
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_ARCHITECTURES 89)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

ament_auto_add_library(imu_processor SHARED
  src/imu_alignment.cpp
  src/imu_dead_reckoning.cpp
  src/imu_processor.cpp
)
target_link_libraries(imu_processor Sophus::Sophus)
if(BACK_ODOM_ENABLE_SIMD AND BACK_ODOM_SIMD_FLAGS)
  target_compile_options(imu_processor PRIVATE ${BACK_ODOM_SIMD_FLAGS})
  target_compile_definitions(imu_processor PRIVATE ${BACK_ODOM_SIMD_DEFINES})
endif()

ament_auto_add_library(lidar_imu_matcher SHARED
  src/lidar_preprocess.cpp
  src/lidar_imu_matcher.cpp
  src/scan_window.cpp
)
target_link_libraries(
  lidar_imu_matcher imu_processor kiss_icp_cpp::kiss_icp_cpp TBB::tbb OpenMP::OpenMP_CXX)
target_sources(lidar_imu_matcher PRIVATE src/device_accel.cpp)
if(BACK_ODOM_ENABLE_SIMD AND BACK_ODOM_SIMD_FLAGS)
  target_compile_options(lidar_imu_matcher PRIVATE ${BACK_ODOM_SIMD_FLAGS})
  target_compile_definitions(lidar_imu_matcher PRIVATE ${BACK_ODOM_SIMD_DEFINES})
endif()
if(CMAKE_CUDA_COMPILER)
  # Own target so nvcc's -Werror all-warnings does not see GNU line markers.
  # The C++ library keeps warnings as errors.
  add_library(back_odom_cuda STATIC src/device_kernels.cu)
  target_include_directories(
    back_odom_cuda PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  set_target_properties(
    back_odom_cuda
    PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON
               COMPILE_WARNING_AS_ERROR OFF)
  target_compile_options(back_odom_cuda PRIVATE -Wno-pedantic)
  target_link_libraries(back_odom_cuda PUBLIC CUDA::cudart)
  target_link_libraries(lidar_imu_matcher back_odom_cuda)
else()
  target_sources(lidar_imu_matcher PRIVATE src/device_kernels_stub.cpp)
endif()

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/back_odom_node.cpp
)
if(BACK_ODOM_ENABLE_SIMD AND BACK_ODOM_SIMD_FLAGS)
  target_compile_options(${PROJECT_NAME} PRIVATE ${BACK_ODOM_SIMD_FLAGS})
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${BACK_ODOM_SIMD_DEFINES})
endif()

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "back_odom::BackOdomNode"
  EXECUTABLE ${PROJECT_NAME}_node
  EXECUTOR SingleThreadedExecutor
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  ament_add_gtest(test_imu_processing test/test_imu_processing.cpp)
  target_link_libraries(test_imu_processing imu_processor)
  ament_add_gtest(test_lidar_match test/test_lidar_match.cpp)
  target_link_libraries(test_lidar_match lidar_imu_matcher)
  ament_add_gtest(test_device_accel test/test_device_accel.cpp)
  target_link_libraries(test_device_accel lidar_imu_matcher)
  ament_add_gtest(test_ekf_messages test/test_ekf_messages.cpp)
  target_link_libraries(test_ekf_messages ${PROJECT_NAME})
  ament_target_dependencies(test_ekf_messages rclcpp tf2_ros sensor_msgs geometry_msgs)
  if(BACK_ODOM_ENABLE_SIMD AND BACK_ODOM_SIMD_FLAGS)
    foreach(_back_odom_test IN ITEMS test_imu_processing test_lidar_match test_device_accel test_ekf_messages)
      target_compile_options(${_back_odom_test} PRIVATE ${BACK_ODOM_SIMD_FLAGS})
      target_compile_definitions(${_back_odom_test} PRIVATE ${BACK_ODOM_SIMD_DEFINES})
    endforeach()
  endif()

  find_package(ament_cmake_pytest QUIET)
  if(ament_cmake_pytest_FOUND)
    ament_add_pytest_test(test_trajectory_metrics test/test_trajectory_metrics.py)
  else()
    add_test(
      NAME test_trajectory_metrics
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_trajectory_metrics.py)
  endif()
endif()

install(
  PROGRAMS
    scripts/eval_trajectory.py
    scripts/dump_odom_csv.py
  DESTINATION lib/${PROJECT_NAME}
)
install(
  FILES scripts/trajectory_metrics.py
  DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY docs/
  DESTINATION share/${PROJECT_NAME}/docs
)

ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
)

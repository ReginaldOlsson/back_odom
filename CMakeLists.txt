cmake_minimum_required(VERSION 3.22)
project(back_odom)

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(Sophus REQUIRED)
find_package(TBB REQUIRED)
find_package(kiss_icp_cpp REQUIRED)
find_package(OpenMP REQUIRED)

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

ament_auto_add_library(lidar_imu_matcher SHARED
  src/lidar_preprocess.cpp
  src/lidar_imu_matcher.cpp
  src/scan_window.cpp
)
target_link_libraries(
  lidar_imu_matcher imu_processor kiss_icp_cpp::kiss_icp_cpp TBB::tbb OpenMP::OpenMP_CXX)
target_sources(lidar_imu_matcher PRIVATE src/device_accel.cpp)
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

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "back_odom::BackOdomNode"
  EXECUTABLE ${PROJECT_NAME}_node
  EXECUTOR SingleThreadedExecutor
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_imu_processing test/test_imu_processing.cpp)
  target_link_libraries(test_imu_processing imu_processor)
  ament_add_gtest(test_lidar_match test/test_lidar_match.cpp)
  target_link_libraries(test_lidar_match lidar_imu_matcher)
  ament_add_gtest(test_device_accel test/test_device_accel.cpp)
  target_link_libraries(test_device_accel lidar_imu_matcher)
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
)

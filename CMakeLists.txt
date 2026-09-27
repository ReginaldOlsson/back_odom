cmake_minimum_required(VERSION 3.22)
project(back_odom)

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(Sophus REQUIRED)

ament_auto_add_library(imu_processor SHARED
  src/imu_alignment.cpp
  src/imu_dead_reckoning.cpp
  src/imu_processor.cpp
)
target_link_libraries(imu_processor Sophus::Sophus)

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
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
)

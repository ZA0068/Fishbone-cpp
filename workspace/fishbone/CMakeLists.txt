cmake_minimum_required(VERSION 3.8)
project(fishbone)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ==========================================
# 1. FIND DEPENDENCIES
# ==========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(image_transport REQUIRED)

find_package(CURL REQUIRED)

# g2o: SE3 pose-graph optimizer for loop closure / global map (VertexSE3/EdgeSE3,
# Levenberg-Marquardt). Provided by the ros-jazzy-libg2o system package.
find_package(g2o REQUIRED)

# --- GLOBAL INCLUDE FIX ---
# This forces the compiler to see cv_bridge headers regardless of target linking
if(cv_bridge_INCLUDE_DIRS)
  include_directories(${cv_bridge_INCLUDE_DIRS})
else()
  # Fallback: Explicitly look in standard ROS include path if variable is empty
  include_directories("/opt/ros/jazzy/include")
endif()

include_directories(include)

# ==========================================
# 2. GENERATE CUSTOM MESSAGES
# ==========================================
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/IMUData.msg"
  "msg/GNSSData.msg"
  "msg/LidarPoint.msg"
  "msg/LidarData.msg"
  "msg/CameraData.msg"
  "msg/StereoCameraData.msg"
  "msg/Frame.msg"
  "msg/PoseData.msg"
  DEPENDENCIES geometry_msgs sensor_msgs
)

# ==========================================
# 3. BUILD NODES (MACRO SOLUTION)
# ==========================================
macro(build_fishbone_node node_name source_file)
  add_executable(${node_name} 
    src/core/${source_file}
    src/nodes/${node_name}.cpp
  )

  target_include_directories(${node_name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
  )

  ament_target_dependencies(${node_name} 
    rclcpp 
    sensor_msgs 
    std_msgs 
    cv_bridge
    pcl_conversions
    geometry_msgs
    OpenCV
    message_filters
  )

  rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
  
  target_link_libraries(${node_name} 
    "${cpp_typesupport_target}"
    ${OpenCV_LIBRARIES} 
    Eigen3::Eigen
    ${CURL_LIBRARIES}
    cv_bridge::cv_bridge
  )

  install(TARGETS ${node_name} DESTINATION lib/${PROJECT_NAME})
endmacro()

# --------------------------------------------------------
# DEFINE SIMPLE NODES
# --------------------------------------------------------
build_fishbone_node(camera_node "camera_main.cpp")
build_fishbone_node(videostream_node "videostream_main.cpp")
build_fishbone_node(frame_builder_node "frame_builder_main.cpp")
build_fishbone_node(lidar_node "lidar_main.cpp")

# --------------------------------------------------------
# DEFINE FISHBONE NODE (The VSLAM "Brain")
# --------------------------------------------------------
add_executable(fishbone_node
  src/core/fishbone_main.cpp
  src/nodes/fishbone_node.cpp
  src/vslam/fovean.cpp
  src/vslam/fishbone.cpp
  src/vslam/soft2_motion.cpp
  src/vslam/rigid_fit.cpp
  src/vslam/pose_graph.cpp
  src/vslam/loop_closure.cpp
)

target_include_directories(fishbone_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS} # Explicit add
)

ament_target_dependencies(fishbone_node 
  rclcpp 
  sensor_msgs 
  nav_msgs
  std_msgs 
  cv_bridge
  geometry_msgs
  OpenCV
  message_filters
  image_transport
  tf2
  tf2_geometry_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")

target_link_libraries(fishbone_node
  "${cpp_typesupport_target}"
  ${OpenCV_LIBRARIES}
  Eigen3::Eigen
  ${CURL_LIBRARIES}
  cv_bridge::cv_bridge
  g2o::core
  g2o::stuff
  g2o::types_slam3d
  g2o::solver_eigen
)

install(TARGETS fishbone_node DESTINATION lib/${PROJECT_NAME})

# ==========================================
# 4. INSTALLATION (Assets)
# ==========================================
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY msg/ DESTINATION share/${PROJECT_NAME}/msg)
install(DIRECTORY rviz/ DESTINATION share/${PROJECT_NAME}/rviz)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(PROGRAMS scripts/kitti_rtabmap_bridge.py DESTINATION lib/${PROJECT_NAME})

# ==========================================
# 5. EXPORT & PACKAGE
# ==========================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
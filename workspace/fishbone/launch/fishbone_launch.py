"""Launches the fishbone VSLAM pipeline against a KITTI sequence.

All tunables (foveation sigmas, matching/PnP thresholds, sequence number,
etc.) come from config/config.yaml -- edit that file and re-run, no rebuild
needed, since these are runtime ROS 2 parameters, not compiled-in constants.
Pass config_file:=/path/to/other.yaml to use a different one.

Unlike the old XML launch file, this one also shuts the whole pipeline down
automatically once camera_node reaches the end of the sequence and exits
(see camera_node.cpp's rclcpp::shutdown() call) -- rviz and the feature
window close with it, instead of needing Ctrl+C.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown, LogInfo
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_rtabmap_fallback = LaunchConfiguration('use_rtabmap_fallback')
    config_file = LaunchConfiguration('config_file')

    # camera_node always runs -- it's the shared image source for both paths.
    camera_node = Node(
        package='fishbone', executable='camera_node', name='camera_node',
        output='screen', parameters=[config_file],
    )

    # fishbone's own pipeline (the "fovean" path) and the RTAB-Map fallback are
    # mutually exclusive, not layered: running both at once means they compete
    # for the same CPU, which is exactly what degraded RTAB-Map's own tracking
    # quality in earlier comparison testing (dropped frames, sync warnings).
    lidar_node = Node(
        package='fishbone', executable='lidar_node', name='lidar_node', output='screen',
        condition=UnlessCondition(use_rtabmap_fallback),
    )
    frame_builder_node = Node(
        package='fishbone', executable='frame_builder_node', name='frame_builder_node',
        output='screen', parameters=[config_file],
        condition=UnlessCondition(use_rtabmap_fallback),
    )
    fishbone_node = Node(
        package='fishbone', executable='fishbone_node', name='fishbone_node',
        output='screen', parameters=[config_file],
        condition=UnlessCondition(use_rtabmap_fallback),
    )
    # NOTE: fishbone.rviz only has displays for fishbone_node's own topics
    # (/fishbone/path_estimated, /fishbone/path_ground_truth, /fishbone/features_image),
    # so it's only useful in the fishbone path -- nothing populates it when
    # use_rtabmap_fallback:=true, so it's skipped there rather than showing an
    # empty/broken window.
    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        arguments=['-d', PathJoinSubstitution([FindPackageShare('fishbone'), 'rviz', 'fishbone.rviz'])],
        condition=UnlessCondition(use_rtabmap_fallback),
    )

    # Fallback path (off by default) -- see use_rtabmap_fallback.
    kitti_rtabmap_bridge = Node(
        package='fishbone', executable='kitti_rtabmap_bridge.py', name='kitti_rtabmap_bridge',
        output='screen', condition=IfCondition(use_rtabmap_fallback),
    )
    stereo_odometry = Node(
        package='rtabmap_odom', executable='stereo_odometry', name='stereo_odometry',
        output='screen', condition=IfCondition(use_rtabmap_fallback),
        remappings=[
            ('left/image_rect', '/kitti/left/image_rect'),
            ('right/image_rect', '/kitti/right/image_rect'),
            ('left/camera_info', '/kitti/left/camera_info'),
            ('right/camera_info', '/kitti/right/camera_info'),
            ('odom', '/rtabmap/odom'),
        ],
        parameters=[{
            'frame_id': 'camera_left',
            'odom_frame_id': 'odom',
            'publish_tf': False,
            'approx_sync': True,
            'Odom/Strategy': '0',
            'Vis/EstimationType': '1',
        }],
    )

    # Once camera_node exits (end of sequence reached, see camera_node.cpp), tear
    # down the whole launch -- this is what actually closes rviz and the feature
    # window instead of requiring Ctrl+C.
    shutdown_on_camera_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=camera_node,
            on_exit=[
                LogInfo(msg='camera_node reached the end of the sequence -- shutting down the pipeline.'),
                Shutdown(reason='end of sequence'),
            ],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=PathJoinSubstitution([FindPackageShare('fishbone'), 'config', 'config.yaml']),
            description='YAML file of ROS 2 parameters for camera_node, frame_builder_node, and fishbone_node.',
        ),
        DeclareLaunchArgument(
            'use_rtabmap_fallback', default_value='false',
            description="RTAB-Map is a fallback for when fishbone's own VSLAM isn't "
                        'producing usable results, disabled by default.',
        ),
        camera_node,
        lidar_node,
        frame_builder_node,
        fishbone_node,
        rviz_node,
        kitti_rtabmap_bridge,
        stereo_odometry,
        shutdown_on_camera_exit,
    ])

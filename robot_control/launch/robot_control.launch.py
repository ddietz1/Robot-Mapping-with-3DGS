
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    map_yaml = LaunchConfiguration('map')

    # stretch_driver_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(os.path.join(
    #         get_package_share_directory('stretch_core'),
    #         'launch', 'stretch_driver.launch.py')),
    # )
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('stretch_core'),
            'launch', 'multi_camera.launch.py'
        )),

    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('stretch_nav2'),
            'launch', 'navigation.launch.py')),
        launch_arguments={'map': map_yaml, 'use_rviz': 'false'}.items()
    )

    move_joints_node = Node(
        package='robot_control',
        executable='move_joints',
        name='move_joints',
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('map', description='Full path to map yaml file'),
        # stretch_driver_launch,
        nav2_launch,
        camera_launch,
        move_joints_node,
    ])
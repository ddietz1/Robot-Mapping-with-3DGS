
import os
import launch_ros
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    map_yamlPath = os.path.join(
        get_package_share_directory('robot_control'),
        'config/maps',
        'MSR_lab.yaml'
    )
    yamlPath = os.path.join(
        get_package_share_directory('robot_control'),
        'config',
        'parameters.yaml'
    )

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
        launch_arguments={'map': map_yamlPath, 'use_rviz': 'false'}.items()
    )

    move_joints_node = Node(
        package='robot_control',
        executable='move_joints',
        name='move_joints',
        output='screen',
        parameters=[yamlPath]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value='/home/hello-robot/stretch_user/maps/MSR_lab.yaml',
            description='Full path to map yaml file'
        ),
        # stretch_driver_launch,
        nav2_launch,
        camera_launch,
        move_joints_node,
    ])
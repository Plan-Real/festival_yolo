import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    params = LaunchConfiguration(
        "festival_yolo_params",
        default=os.path.join(
            get_package_share_directory(
                "festival_yolo"), "configs", "festival_yolo.yaml"
        ),
    )
    params_cmd = DeclareLaunchArgument(
        "festival_yolo_params",
        default_value=params,
        description="Path to the ROS2 parameters file to use for all nodes",
    )
    model_path = os.path.join(
        get_package_share_directory(
            "festival_yolo"), "checkpoints"
    )


    #
    # NODES
    #
    yolo_node_cmd = Node(
        package="festival_yolo",
        executable="yolo",
        name="festival_yolo_node",
        parameters=[params],
        output="screen",
    )

    ld = LaunchDescription()

    ld.add_action(params_cmd)
    ld.add_action(yolo_node_cmd)

    return ld

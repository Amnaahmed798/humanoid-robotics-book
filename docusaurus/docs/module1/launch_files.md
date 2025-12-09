# Launch Files and Parameter Management

## Introduction to Launch Files

Launch files in ROS 2 allow you to start multiple nodes at once with a single command. They provide a way to configure and run complex systems with many interconnected nodes. Launch files can also manage parameters, remappings, and other configuration options.

## Launch File Structure

Launch files are typically written in Python using the `launch` package. Here's the basic structure:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='package_name',
            executable='executable_name',
            name='node_name',
            parameters=[
                {'param_name': 'param_value'},
                'path/to/params.yaml'
            ],
            remappings=[
                ('original_topic', 'new_topic')
            ]
        )
    ])
```

## Creating Launch Files

### Basic Node Launch

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='sim'
        ),
        Node(
            package='turtlesim',
            executable='turtle_teleop_key',
            name='teleop'
        )
    ])
```

### Launch File with Parameters

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    param_file = LaunchConfiguration('param_file')
    declare_param_file_arg = DeclareLaunchArgument(
        'param_file',
        default_value='path/to/params.yaml',
        description='Path to parameters file'
    )

    return LaunchDescription([
        declare_param_file_arg,
        Node(
            package='my_package',
            executable='my_node',
            name='my_node',
            parameters=[param_file]
        )
    ])
```

## Parameter Management

### YAML Parameter Files

Parameters can be stored in YAML files for easy management:

```yaml
my_node:
  ros__parameters:
    param1: value1
    param2: 42
    param3: true
    nested_param:
      sub_param1: "nested_value"
      sub_param2: 3.14
```

### Using Parameters in Nodes

In your node, you can access parameters like this:

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')

        # Declare parameters with default values
        self.declare_parameter('param1', 'default_value')
        self.declare_parameter('param2', 0)

        # Get parameter values
        param1_value = self.get_parameter('param1').value
        param2_value = self.get_parameter('param2').value

        self.get_logger().info(f'param1: {param1_value}, param2: {param2_value}')
```

## Advanced Launch Features

### Conditional Launch

```python
from launch import LaunchDescription, LaunchCondition
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    return LaunchDescription([
        declare_use_sim_time_arg,
        Node(
            condition=LaunchCondition(use_sim_time),
            package='my_package',
            executable='sim_node',
            name='sim_node'
        )
    ])
```

### Including Other Launch Files

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('other_package'),
                    'launch',
                    'other_launch_file.py'
                ])
            ])
        )
    ])
```

## Best Practices

1. Use launch files to start complete systems rather than individual nodes
2. Organize parameters in YAML files for easy configuration
3. Use launch arguments to make launch files configurable
4. Group related nodes in the same launch file
5. Use appropriate namespaces for nodes to avoid naming conflicts
6. Include error handling and logging in launch files
7. Document launch arguments and their purposes
8. Use standard parameter naming conventions
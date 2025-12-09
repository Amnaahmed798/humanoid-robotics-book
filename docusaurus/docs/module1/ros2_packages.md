# ROS 2 Package Structure

## Introduction

ROS 2 packages are the fundamental building blocks of ROS 2 software. A package contains libraries, executables, configuration files, and other resources needed for a specific task. Understanding the structure of ROS 2 packages is crucial for organizing your robot software effectively.

## Package Structure

A typical ROS 2 package has the following structure:

```
my_robot_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml             # Package metadata
├── setup.py                # Build configuration for Python
├── setup.cfg               # Installation configuration
├── src/                    # Source code (C++)
│   └── my_node.cpp
├── my_robot_package/       # Python modules
│   ├── __init__.py
│   └── my_module.py
├── launch/                 # Launch files
│   └── my_launch_file.py
├── config/                 # Configuration files
│   └── params.yaml
├── urdf/                   # Robot description files
│   └── robot.urdf
├── meshes/                 # 3D mesh files
│   └── part.stl
├── include/                # Header files (C++)
│   └── my_robot_package/
│       └── my_header.h
└── test/                   # Test files
    └── test_my_module.py
```

## package.xml

The `package.xml` file contains metadata about the package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Package for my robot functionality</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## setup.py

For Python packages, `setup.py` defines the build configuration:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Include URDF files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='user@example.com',
    description='Package for my robot functionality',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = my_robot_package.my_node:main',
        ],
    },
)
```

## Python Node Structure

A typical Python node in a ROS 2 package follows this structure:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from my_robot_package.my_library import MyHelperClass

class MyRobotNode(Node):
    def __init__(self):
        super().__init__('my_robot_node')

        # Create subscribers
        self.subscription = self.create_subscription(
            String,
            'topic_name',
            self.listener_callback,
            10
        )

        # Create publishers
        self.publisher = self.create_publisher(
            JointState,
            'joint_states',
            10
        )

        # Create timers
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Declare parameters
        self.declare_parameter('param_name', 'default_value')

        self.get_logger().info('MyRobotNode has been started')

    def listener_callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')

    def timer_callback(self):
        msg = JointState()
        # Populate message
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MyRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating a New Package

### Using colcon

```bash
# Create a new workspace
mkdir -p ~/ros2_workspace/src
cd ~/ros2_workspace/src

# Create a new package (Python)
ros2 pkg create --build-type ament_python my_robot_package

# Create a new package (C++)
ros2 pkg create --build-type ament_cmake my_robot_package
```

### Manual Creation

For Python packages, create the structure manually:

```bash
mkdir -p my_robot_package/my_robot_package
touch my_robot_package/my_robot_package/__init__.py
```

## Launch Files in Packages

Launch files should be placed in the `launch/` directory:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('my_robot_package')

    # Include URDF file
    urdf_file = os.path.join(pkg_share, 'urdf', 'robot.urdf')

    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='my_node',
            name='my_robot_node',
            parameters=[
                {'param_name': 'param_value'},
                os.path.join(pkg_share, 'config', 'params.yaml')
            ]
        )
    ])
```

## Configuration Files

Configuration files should be placed in the `config/` directory:

```yaml
# params.yaml
my_robot_node:
  ros__parameters:
    rate: 10
    enable_feature: true
    threshold: 0.5
    topic_name: "/my_topic"
```

## Best Practices

1. Use descriptive and consistent naming conventions
2. Group related functionality in the same package
3. Keep packages focused on a single responsibility
4. Use appropriate build types (ament_python for Python, ament_cmake for C++)
5. Include proper documentation in package.xml
6. Add dependencies explicitly in package.xml
7. Organize files in appropriate subdirectories
8. Use the same name for the package and main Python module
9. Include proper licensing information
10. Test your package with `colcon build` and `ros2 run`
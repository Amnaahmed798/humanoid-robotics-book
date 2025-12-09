# Gazebo Simulation Environment Setup

## Introduction

Gazebo is a powerful, open-source robotics simulator that provides accurate physics simulation, high-quality graphics, and convenient programmatic interfaces. It's widely used in robotics research and development for testing algorithms, validating robot designs, and training AI models.

## Installing Gazebo

### For ROS 2 Humble Hawksbill (Ubuntu 22.04)

```bash
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins ros-humble-gazebo-dev
```

### For Other ROS 2 Distributions

Replace `humble` with your ROS 2 distribution name (e.g., `foxy`, `galactic`, `rolling`).

## Launching Gazebo

### Basic Gazebo Launch

```bash
# Launch Gazebo with an empty world
ros2 launch gazebo_ros empty_world.launch.py

# Launch Gazebo with a specific world
ros2 launch gazebo_ros empty_world.launch.py world_name:=my_world.sdf
```

### Launching with Custom Parameters

```bash
# Launch with GUI disabled (headless)
ros2 launch gazebo_ros empty_world.launch.py gui_required:=false

# Launch with verbose output
ros2 launch gazebo_ros empty_world.launch.py verbose:=true
```

## Basic Gazebo Concepts

### Worlds

World files define the environment in which robots operate. They contain:
- Physical properties (gravity, atmosphere)
- Models and their initial positions
- Lighting and visual properties
- Plugins for custom behavior

### Models

Models represent objects in the simulation:
- Robots
- Static objects (tables, walls, etc.)
- Dynamic objects (balls, boxes, etc.)

### Plugins

Plugins extend Gazebo's functionality:
- Physics plugins
- Sensor plugins
- Control plugins
- GUI plugins

## Creating a Basic World File

Here's a simple world file example:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="basic_world">
    <!-- Include a model from Gazebo's model database -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Define a simple box model -->
    <model name="simple_box">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="box_link">
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.083</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.083</iyy>
            <iyz>0</iyz>
            <izz>0.083</izz>
          </inertia>
        </inertial>
        <collision name="box_collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="box_visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Spawning Robots in Gazebo

### Using Command Line

```bash
# Spawn a robot from a URDF file
ros2 run gazebo_ros spawn_entity.py -entity my_robot -file /path/to/robot.urdf -x 0 -y 0 -z 1

# Spawn with custom position and orientation
ros2 run gazebo_ros spawn_entity.py -entity my_robot -file /path/to/robot.urdf -x 1 -y 2 -z 0.5 -R 0 -P 0 -Y 1.57
```

### Using Launch Files

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'empty_world.launch.py'
            ])
        ])
    )

    # Spawn robot
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'my_robot',
            '-file', '/path/to/robot.urdf',
            '-x', '0', '-y', '0', '-z', '0.5'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        spawn_entity
    ])
```

## Working with Gazebo Models

### Finding Models

Gazebo includes many pre-built models. You can find them at:
- `/usr/share/gazebo/models` (system installation)
- `~/.gazebo/models` (user models)

### Creating Custom Models

A custom model directory structure:

```
my_model/
├── model.config
└── model.sdf
```

**model.config**:
```xml
<?xml version="1.0"?>
<model>
  <name>My Custom Model</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>A custom model for my robot.</description>
</model>
```

## Connecting to Gazebo from ROS 2

### Gazebo ROS Packages

The `gazebo_ros_pkgs` package provides bridges between ROS 2 and Gazebo:

- `libgazebo_ros_factory`: Spawn and delete entities
- `libgazebo_ros_init`: Initialize ROS 2 within Gazebo
- `libgazebo_ros_force`: Apply forces and torques to models

### Basic Connection

```python
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from gazebo_msgs.msg import ModelStates

class GazeboInterface(Node):
    def __init__(self):
        super().__init__('gazebo_interface')

        # Service clients for spawning/deleting entities
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')

        # Subscriber for model states
        self.model_state_sub = self.create_subscription(
            ModelStates,
            '/model_states',
            self.model_state_callback,
            10
        )

    def model_state_callback(self, msg):
        # Process model states
        for i, name in enumerate(msg.name):
            if name == 'my_robot':
                position = msg.pose[i].position
                self.get_logger().info(f'Robot position: {position}')
```

## Best Practices

1. **Start Simple**: Begin with basic worlds and gradually add complexity
2. **Use Standard Models**: Leverage existing models when possible
3. **Validate Physics**: Ensure your models have realistic mass and inertia properties
4. **Optimize Performance**: Adjust physics parameters for simulation speed vs. accuracy
5. **Test Incrementally**: Test each component before integrating
6. **Document Worlds**: Include comments and documentation in world files
7. **Version Control**: Keep world files under version control alongside your robot code
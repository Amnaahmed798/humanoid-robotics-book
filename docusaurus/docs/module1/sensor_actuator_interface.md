# Interfacing Sensors and Actuators

## Introduction

In robotics, sensors provide information about the robot's state and environment, while actuators allow the robot to interact with the world. Properly interfacing sensors and actuators is crucial for robot functionality. This chapter covers how to work with various sensors and actuators in ROS 2.

## Sensor Types and Messages

### Common Sensor Types

1. **IMU (Inertial Measurement Unit)**: Measures orientation, angular velocity, and linear acceleration
   - Message type: `sensor_msgs/Imu`
   - Topics: `/imu/data`, `/imu/raw`

2. **LIDAR**: Measures distances using laser light
   - Message type: `sensor_msgs/LaserScan` or `sensor_msgs/PointCloud2`
   - Topics: `/scan`, `/points`

3. **Cameras**: Visual sensors for image processing
   - Message type: `sensor_msgs/Image` or `sensor_msgs/CompressedImage`
   - Topics: `/camera/image_raw`, `/camera/compressed`

4. **Joint Sensors**: Provide joint position, velocity, and effort
   - Message type: `sensor_msgs/JointState`
   - Topic: `/joint_states`

5. **Force/Torque Sensors**: Measure forces and torques
   - Message type: `geometry_msgs/Wrench`
   - Topics: `/wrench`, `/ft_sensor`

### Sensor Message Structure

```python
# Example of JointState message
from sensor_msgs.msg import JointState

def sensor_callback(self, msg):
    # msg.name: list of joint names
    # msg.position: list of joint positions (radians)
    # msg.velocity: list of joint velocities (rad/s)
    # msg.effort: list of joint efforts (Nm)

    for i, name in enumerate(msg.name):
        position = msg.position[i]
        velocity = msg.velocity[i] if msg.velocity else None
        effort = msg.effort[i] if msg.effort else None

        self.get_logger().info(f'{name}: pos={position}, vel={velocity}, effort={effort}')
```

## Actuator Types and Control

### Joint Actuators

Joint actuators control the movement of robot joints. In ROS 2, joint control is typically handled through:

1. **Position Control**: Commands specific joint angles
2. **Velocity Control**: Commands specific joint velocities
3. **Effort Control**: Commands specific joint torques

### Joint Trajectory Control

For coordinated multi-joint movements, ROS 2 uses trajectory messages:

```python
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryGoal

class ActuatorController(Node):
    def __init__(self):
        super().__init__('actuator_controller')

        # Publisher for joint trajectory commands
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

    def send_trajectory(self, joint_names, positions, time_from_start=1.0):
        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(time_from_start)
        point.time_from_start.nanosec = int((time_from_start % 1) * 1e9)

        traj_msg.points = [point]
        self.traj_pub.publish(traj_msg)
```

## Hardware Interface

### Creating a Sensor Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import random

class SensorSimulator(Node):
    def __init__(self):
        super().__init__('sensor_simulator')

        # Create publisher for joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Timer to publish sensor data at regular intervals
        self.timer = self.create_timer(0.1, self.publish_joint_states)

        # Initialize joint names
        self.joint_names = ['joint1', 'joint2', 'joint3']

    def publish_joint_states(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = [random.uniform(-1.0, 1.0) for _ in self.joint_names]
        msg.velocity = [random.uniform(-0.5, 0.5) for _ in self.joint_names]
        msg.effort = [random.uniform(-10.0, 10.0) for _ in self.joint_names]

        # Set timestamp
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        self.joint_pub.publish(msg)
```

### Creating an Actuator Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

class ActuatorController(Node):
    def __init__(self):
        super().__init__('actuator_controller')

        # Subscriber for commands
        self.command_sub = self.create_subscription(
            Float64MultiArray,
            'joint_commands',
            self.command_callback,
            10
        )

        # Subscriber for current joint states
        self.state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.state_callback,
            10
        )

        # Publisher for actuator commands
        self.actuator_pub = self.create_publisher(
            Float64MultiArray,
            'actuator_commands',
            10
        )

        self.current_positions = {}

    def state_callback(self, msg):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_positions[name] = msg.position[i]

    def command_callback(self, msg):
        # Process command and send to actuators
        command = Float64MultiArray()
        command.data = self.process_command(msg.data)
        self.actuator_pub.publish(command)

    def process_command(self, target_positions):
        # Implement control logic here
        # This could include PID control, filtering, etc.
        return target_positions  # Simplified
```

## Real Hardware Interface

### Using ros2_control

The `ros2_control` framework provides a standardized way to interface with hardware:

```python
from controller_manager_msgs.srv import SwitchController
from hardware_interface import HARDWARE_INTERFACE_VERSION

class HardwareInterface(Node):
    def __init__(self):
        super().__init__('hardware_interface')

        # Service client for controller switching
        self.switch_client = self.create_client(
            SwitchController,
            '/controller_manager/switch_controller'
        )

    def switch_controllers(self, start_controllers, stop_controllers):
        req = SwitchController.Request()
        req.start_controllers = start_controllers
        req.stop_controllers = stop_controllers
        req.strictness = SwitchController.Request.BEST_EFFORT

        future = self.switch_client.call_async(req)
        return future
```

## Sensor Fusion

For better state estimation, multiple sensors can be fused:

```python
from robot_localization.srv import FromLL, ToLL
from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import PoseWithCovarianceStamped

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Subscribers for different sensor types
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, 10)
        self.gps_sub = self.create_subscription(NavSatFix, 'gps/fix', self.gps_callback, 10)

        # Publisher for fused pose
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'fused_pose', 10)

        # Initialize filter (e.g., Extended Kalman Filter)
        self.initialize_filter()

    def initialize_filter(self):
        # Initialize your chosen filtering algorithm
        pass

    def imu_callback(self, msg):
        # Process IMU data and update filter
        pass

    def gps_callback(self, msg):
        # Process GPS data and update filter
        pass
```

## Safety Considerations

### Limits and Constraints

```python
class SafeActuatorController(Node):
    def __init__(self):
        super().__init__('safe_actuator_controller')

        # Define safety limits
        self.position_limits = {
            'joint1': (-1.5, 1.5),
            'joint2': (-2.0, 2.0),
            'joint3': (-3.0, 3.0)
        }

        self.velocity_limit = 1.0  # rad/s
        self.effort_limit = 50.0   # Nm

    def check_limits(self, positions, velocities, efforts):
        for i, (pos, vel, eff) in enumerate(zip(positions, velocities, efforts)):
            joint_name = f'joint{i+1}'

            # Check position limits
            if joint_name in self.position_limits:
                min_pos, max_pos = self.position_limits[joint_name]
                if not (min_pos <= pos <= max_pos):
                    self.get_logger().warn(f'Position limit exceeded for {joint_name}')
                    return False

            # Check velocity limits
            if abs(vel) > self.velocity_limit:
                self.get_logger().warn(f'Velocity limit exceeded for {joint_name}')
                return False

            # Check effort limits
            if abs(eff) > self.effort_limit:
                self.get_logger().warn(f'Effort limit exceeded for {joint_name}')
                return False

        return True
```

## Best Practices

1. **Use standard message types** when possible to ensure compatibility
2. **Implement proper error handling** for sensor failures
3. **Apply appropriate filtering** to sensor data to reduce noise
4. **Implement safety limits** to protect hardware
5. **Use appropriate QoS settings** for real-time requirements
6. **Validate sensor data** before using in control algorithms
7. **Implement graceful degradation** when sensors fail
8. **Log sensor and actuator data** for debugging and analysis
9. **Consider timing constraints** for real-time control
10. **Test with simulated hardware** before deploying to real robots
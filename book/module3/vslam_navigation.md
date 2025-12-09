# Visual SLAM (VSLAM) and Navigation using Isaac ROS

## Introduction

Visual Simultaneous Localization and Mapping (VSLAM) is a critical capability for autonomous robots, allowing them to build maps of their environment while simultaneously determining their position within that map. NVIDIA Isaac ROS provides optimized VSLAM implementations that leverage GPU acceleration for real-time performance. This chapter covers the theory and practical implementation of VSLAM using Isaac ROS.

## Understanding VSLAM

### What is VSLAM?

VSLAM combines computer vision and robotics to solve two problems simultaneously:
1. **Mapping**: Creating a map of the environment
2. **Localization**: Determining the robot's position within the map

### Key Components of VSLAM

1. **Feature Detection**: Identifying distinctive points in images
2. **Feature Matching**: Finding corresponding features across frames
3. **Pose Estimation**: Calculating the robot's position and orientation
4. **Map Building**: Creating a consistent representation of the environment
5. **Loop Closure**: Recognizing previously visited locations to correct drift

## Isaac ROS Visual SLAM

### Overview

Isaac ROS provides optimized VSLAM implementations including:
- **Isaac ROS Visual SLAM**: GPU-accelerated visual SLAM
- **Isaac ROS Stereo Visual SLAM**: Stereo camera-based SLAM
- **Isaac ROS Occupancy Grid Localizer**: Grid-based localization

### Key Features

- **GPU Acceleration**: Leverages CUDA for real-time performance
- **Multi-camera Support**: Supports monocular, stereo, and multi-camera setups
- **ROS 2 Integration**: Seamless integration with ROS 2 ecosystem
- **Real-time Performance**: Optimized for robotic applications

## Setting up Isaac ROS Visual SLAM

### Prerequisites

Before using Isaac ROS Visual SLAM, ensure you have:
- NVIDIA GPU with Compute Capability 6.0+
- Isaac ROS packages installed
- Camera calibrated with ROS camera_info

### Installation and Dependencies

The Isaac ROS Visual SLAM package includes:
- `isaac_ros_visual_slam`: Main VSLAM package
- `isaac_ros_freespace_segmentation`: Free space detection
- `isaac_ros_gxf`: GXF (GEMS eXtensible Framework) extensions

### Basic Launch Configuration

```xml
<!-- visual_slam_launch.xml -->
<launch>
  <!-- Visual SLAM node -->
  <node pkg="isaac_ros_visual_slam" exec="visual_slam_node" name="visual_slam_node" output="screen">
    <param name="enable_rectified_pose" value="true"/>
    <param name="enable_fisheye" value="false"/>
    <param name="rectified_frame_id" value="camera_link"/>
    <param name="enable_debug_mode" value="false"/>
    <param name="enable_slam_visualization" value="true"/>
  </node>
</launch>
```

### Python Launch File

```python
# visual_slam_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    enable_rectified_pose = LaunchConfiguration('enable_rectified_pose')
    enable_fisheye = LaunchConfiguration('enable_fisheye')
    rectified_frame_id = LaunchConfiguration('rectified_frame_id')

    # Visual SLAM node
    visual_slam_node = Node(
        package='isaac_ros_visual_slam',
        executable='visual_slam_node',
        name='visual_slam_node',
        parameters=[{
            'enable_rectified_pose': enable_rectified_pose,
            'enable_fisheye': enable_fisheye,
            'rectified_frame_id': rectified_frame_id,
        }],
        remappings=[
            ('/visual_slam/image', '/camera/image_raw'),
            ('/visual_slam/camera_info', '/camera/camera_info'),
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('enable_rectified_pose', default_value='true'),
        DeclareLaunchArgument('enable_fisheye', default_value='false'),
        DeclareLaunchArgument('rectified_frame_id', default_value='camera_link'),
        visual_slam_node
    ])
```

## Camera Calibration for VSLAM

### Importance of Calibration

Proper camera calibration is crucial for accurate VSLAM performance:
- Corrects lens distortion
- Provides accurate intrinsic parameters
- Enables accurate 3D reconstruction

### Calibration Process

```bash
# Use ROS camera calibration tool
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.108 image:=/camera/image_raw camera:=/camera

# Or use Isaac Sim calibration tools for synthetic data
```

### Calibration File Format

```yaml
# camera_info.yaml
camera_name: my_camera
image_width: 640
image_height: 480
camera_matrix:
  rows: 3
  cols: 3
  data: [615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0]
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.1, -0.2, 0.0, 0.0, 0.0]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [615.0, 0.0, 320.0, 0.0, 0.0, 615.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
distortion_model: plumb_bob
```

## Implementing VSLAM with Isaac ROS

### Basic VSLAM Node Implementation

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
import numpy as np
from cv_bridge import CvBridge


class IsaacVSLAMInterface(Node):
    """
    Interface to Isaac ROS Visual SLAM with additional processing.
    """

    def __init__(self):
        super().__init__('isaac_vslam_interface')

        # CV Bridge for image processing
        self.bridge = CvBridge()

        # Subscribers for Isaac VSLAM output
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/visual_slam/odometry',
            self.odom_callback,
            10
        )

        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/visual_slam/pose',
            self.pose_callback,
            10
        )

        self.map_subscription = self.create_subscription(
            MarkerArray,
            '/visual_slam/landmarks',
            self.map_callback,
            10
        )

        # Publishers for processed data
        self.robot_path_publisher = self.create_publisher(
            MarkerArray,
            '/robot_path',
            10
        )

        # Internal state
        self.robot_path = []
        self.current_pose = None

        # Parameters
        self.path_max_length = 1000  # Maximum path length to store
        self.path_publish_rate = 10  # Hz

        # Timer for path publishing
        self.path_timer = self.create_timer(
            1.0 / self.path_publish_rate,
            self.publish_path
        )

        self.get_logger().info('Isaac VSLAM Interface initialized')

    def odom_callback(self, msg):
        """Process odometry data from Isaac VSLAM."""
        self.current_pose = msg.pose.pose

        # Add to path if valid
        if self.current_pose is not None:
            self.robot_path.append(self.current_pose)

            # Limit path length
            if len(self.robot_path) > self.path_max_length:
                self.robot_path.pop(0)

        self.get_logger().debug(
            f'VSLAM Odometry: ({msg.pose.pose.position.x:.2f}, '
            f'{msg.pose.pose.position.y:.2f}, {msg.pose.pose.position.z:.2f})'
        )

    def pose_callback(self, msg):
        """Process pose data from Isaac VSLAM."""
        self.get_logger().debug(
            f'VSLAM Pose: ({msg.pose.position.x:.2f}, '
            f'{msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})'
        )

    def map_callback(self, msg):
        """Process landmark data from Isaac VSLAM."""
        self.get_logger().info(f'VSLAM Map: {len(msg.markers)} landmarks detected')

        # Process landmarks (implementation depends on specific use case)
        for marker in msg.markers:
            self.get_logger().debug(
                f'Landmark {marker.id}: '
                f'({marker.pose.position.x:.2f}, {marker.pose.position.y:.2f})'
            )

    def publish_path(self):
        """Publish robot path as visualization markers."""
        if len(self.robot_path) == 0 or self.current_pose is None:
            return

        # Create path visualization (simplified)
        path_marker = MarkerArray()
        # Implementation would create line strip or points for path visualization
        # This is a placeholder for the actual visualization implementation

        self.robot_path_publisher.publish(path_marker)


class VSLAMNavigator(Node):
    """
    Navigation node using VSLAM data for path planning and obstacle avoidance.
    """

    def __init__(self):
        super().__init__('vslam_navigator')

        # Subscribers
        self.vslam_odom_sub = self.create_subscription(
            Odometry,
            '/visual_slam/odometry',
            self.vslam_odom_callback,
            10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Navigation parameters
        self.linear_speed = 0.3  # m/s
        self.angular_speed = 0.5  # rad/s
        self.min_distance_to_goal = 0.5  # meters

        # Goal position (example)
        self.goal_x = 5.0
        self.goal_y = 5.0

        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.get_logger().info('VSLAM Navigator initialized')

    def vslam_odom_callback(self, msg):
        """Update robot position from VSLAM odometry."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Extract yaw from quaternion
        quat = msg.pose.pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        self.current_yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Execute navigation logic
        self.navigate_to_goal()

    def navigate_to_goal(self):
        """Simple navigation to goal using VSLAM position."""
        # Calculate distance and angle to goal
        dx = self.goal_x - self.current_x
        dy = self.goal_y - self.current_y
        distance_to_goal = np.sqrt(dx**2 + dy**2)
        goal_angle = np.arctan2(dy, dx)

        # Check if goal reached
        if distance_to_goal < self.min_distance_to_goal:
            self.get_logger().info('Goal reached!')
            self.stop_robot()
            return

        # Calculate angle difference
        angle_diff = self.normalize_angle(goal_angle - self.current_yaw)

        # Create twist command
        cmd_vel = Twist()

        if abs(angle_diff) > 0.2:  # Need to turn
            cmd_vel.angular.z = self.angular_speed * np.sign(angle_diff)
            cmd_vel.linear.x = 0.0
        else:  # Move forward
            cmd_vel.linear.x = min(self.linear_speed, distance_to_goal)
            cmd_vel.angular.z = 0.0

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range."""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def stop_robot(self):
        """Stop the robot."""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)


def main(args=None):
    rclpy.init(args=args)

    # Create nodes
    vslam_interface = IsaacVSLAMInterface()
    navigator = VSLAMNavigator()

    # Create executor
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(vslam_interface)
    executor.add_node(navigator)

    try:
        executor.spin()
    except KeyboardInterrupt:
        vslam_interface.get_logger().info('Shutting down nodes...')
    finally:
        vslam_interface.destroy_node()
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Stereo VSLAM

### Stereo Camera Setup

Stereo VSLAM uses two synchronized cameras to estimate depth:

```python
class IsaacStereoVSLAM(Node):
    """
    Interface for Isaac ROS Stereo Visual SLAM.
    """

    def __init__(self):
        super().__init__('isaac_stereo_vslam')

        # Stereo camera topics
        self.left_image_sub = self.create_subscription(
            Image,
            '/stereo_camera/left/image_raw',
            self.left_image_callback,
            10
        )

        self.right_image_sub = self.create_subscription(
            Image,
            '/stereo_camera/right/image_raw',
            self.right_image_callback,
            10
        )

        self.left_info_sub = self.create_subscription(
            CameraInfo,
            '/stereo_camera/left/camera_info',
            self.left_info_callback,
            10
        )

        self.right_info_sub = self.create_subscription(
            CameraInfo,
            '/stereo_camera/right/camera_info',
            self.right_info_callback,
            10
        )

        # Internal buffers
        self.left_image = None
        self.right_image = None
        self.left_camera_info = None
        self.right_camera_info = None

        self.get_logger().info('Isaac Stereo VSLAM initialized')

    def left_image_callback(self, msg):
        self.left_image = msg
        self.process_stereo_pair()

    def right_image_callback(self, msg):
        self.right_image = msg
        self.process_stereo_pair()

    def left_info_callback(self, msg):
        self.left_camera_info = msg

    def right_info_callback(self, msg):
        self.right_camera_info = msg

    def process_stereo_pair(self):
        """Process synchronized stereo images."""
        if (self.left_image is not None and
            self.right_image is not None and
            self.left_camera_info is not None and
            self.right_camera_info is not None):

            # Process stereo pair using Isaac ROS stereo VSLAM
            # This would typically involve publishing to Isaac ROS stereo nodes
            pass
```

## Performance Optimization

### GPU Memory Management

```python
class OptimizedVSLAMNode(Node):
    """
    VSLAM node with GPU memory optimization.
    """

    def __init__(self):
        super().__init__('optimized_vslam_node')

        # Set GPU memory fraction if needed
        # This depends on the specific Isaac ROS implementation
        self.gpu_memory_fraction = 0.8  # Use 80% of GPU memory

        # Optimize processing pipeline
        self.frame_skip = 2  # Process every 2nd frame to reduce load
        self.frame_counter = 0

        # Initialize other components
        self.setup_vslam_pipeline()

    def setup_vslam_pipeline(self):
        """Configure VSLAM pipeline for optimal performance."""
        # This would include setting up Isaac ROS VSLAM with optimized parameters
        pass
```

## Evaluation and Validation

### VSLAM Quality Metrics

```python
class VSLAMEvaluator(Node):
    """
    Node to evaluate VSLAM performance.
    """

    def __init__(self):
        super().__init__('vslam_evaluator')

        # Subscribe to VSLAM output
        self.vslam_odom_sub = self.create_subscription(
            Odometry,
            '/visual_slam/odometry',
            self.vslam_odom_callback,
            10
        )

        # Track metrics
        self.initial_pose = None
        self.current_pose = None
        self.path_length = 0.0
        self.previous_position = None
        self.feature_count = 0
        self.tracking_quality = 0.0

        # Ground truth (if available) for comparison
        self.ground_truth_sub = self.create_subscription(
            Odometry,
            '/ground_truth/odometry',
            self.ground_truth_callback,
            10
        )

        self.error_metrics = {
            'position_error': [],
            'orientation_error': [],
            'cumulative_error': 0.0
        }

        # Timer for periodic evaluation
        self.eval_timer = self.create_timer(1.0, self.evaluate_performance)

        self.get_logger().info('VSLAM Evaluator initialized')

    def vslam_odom_callback(self, msg):
        """Process VSLAM odometry for evaluation."""
        self.current_pose = msg.pose.pose

        # Update path length
        if self.previous_position is not None:
            dx = msg.pose.pose.position.x - self.previous_position.x
            dy = msg.pose.pose.position.y - self.previous_position.y
            dz = msg.pose.pose.position.z - self.previous_position.z
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            self.path_length += dist

        self.previous_position = msg.pose.pose.position

    def ground_truth_callback(self, msg):
        """Process ground truth for error calculation."""
        if self.current_pose is not None:
            # Calculate position error
            dx = msg.pose.pose.position.x - self.current_pose.position.x
            dy = msg.pose.pose.position.y - self.current_pose.position.y
            dz = msg.pose.pose.position.z - self.current_pose.position.z
            position_error = np.sqrt(dx**2 + dy**2 + dz**2)

            self.error_metrics['position_error'].append(position_error)

            # Calculate orientation error
            # Implementation would compare quaternions

    def evaluate_performance(self):
        """Evaluate and log VSLAM performance metrics."""
        avg_position_error = np.mean(self.error_metrics['position_error']) if self.error_metrics['position_error'] else 0.0

        self.get_logger().info(
            f'VSLAM Performance - '
            f'Avg Position Error: {avg_position_error:.3f}m, '
            f'Path Length: {self.path_length:.3f}m, '
            f'Tracking Quality: {self.tracking_quality:.2f}'
        )
```

## Best Practices for VSLAM

### Camera Selection and Placement

1. **Field of View**: Choose cameras with appropriate FOV for your application
2. **Resolution**: Balance between detail and computational requirements
3. **Mounting**: Secure mounting to minimize vibration and movement
4. **Baseline**: For stereo, appropriate baseline distance for depth range

### Environmental Considerations

1. **Lighting**: Ensure consistent lighting conditions
2. **Texture**: Environments should have sufficient visual features
3. **Dynamic Objects**: Consider impact of moving objects
4. **Reflective Surfaces**: Minimize highly reflective surfaces

### Performance Optimization

1. **Frame Rate**: Match processing rate to robot dynamics
2. **Feature Density**: Optimize for sufficient but not excessive features
3. **Map Management**: Implement efficient map storage and retrieval
4. **Loop Closure**: Regularly detect and correct for drift

### Troubleshooting Common Issues

1. **Drift**: Implement loop closure and relocalization
2. **Feature Loss**: Ensure adequate lighting and texture
3. **Computational Load**: Optimize parameters for real-time performance
4. **Initialization**: Proper initial pose estimation

Isaac ROS Visual SLAM provides a powerful foundation for robotic navigation and mapping applications, with GPU acceleration enabling real-time performance for complex robotic systems.
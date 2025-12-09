# Capstone Project: Autonomous Humanoid Robot

## Introduction

The capstone project brings together all the concepts learned in this book to create a fully autonomous humanoid robot system. This project integrates ROS 2, simulation environments, AI perception and planning, and multi-modal interaction to create a robot that can understand natural language commands, navigate complex environments, manipulate objects, and interact naturally with humans.

## Project Overview

### Objectives

The autonomous humanoid robot will be capable of:
- Understanding and responding to natural language commands
- Navigating to specified locations in indoor environments
- Detecting and manipulating objects using vision and manipulation
- Engaging in multi-modal interaction (speech, gesture, vision)
- Executing complex, multi-step tasks
- Demonstrating autonomous behavior in realistic scenarios

### Technical Requirements

- **Hardware**: Simulated humanoid robot with 2 arms, mobile base, and sensors
- **Software**: ROS 2 Humble with Isaac ROS packages
- **Sensors**: RGB-D camera, IMU, LiDAR, joint encoders
- **Actuators**: 7-DOF arms, mobile base, grippers
- **AI Components**: LLM for planning, VSLAM for navigation, object recognition

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
├─────────────────────────────────────────────────────────────────┤
│  Natural Language    │  Multi-Modal    │  Visual Interface    │
│  Commands           │  Interaction    │  (RViz, Dashboard)   │
└─────────────┬──────────────────────┬──────────────────┬────────┘
              │                      │                  │
              ▼                      ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     COGNITIVE PLANNING                          │
├─────────────────────────────────────────────────────────────────┤
│  LLM-Based         │  Task Planning    │  Motion Planning     │
│  Command Parser    │  & Sequencing    │  & Trajectory Gen    │
└─────────────┬──────────────────────┬──────────────────┬────────┘
              │                      │                  │
              ▼                      ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXECUTION SYSTEM                              │
├─────────────────────────────────────────────────────────────────┤
│  Navigation        │  Manipulation    │  Perception          │
│  Stack (Nav2)     │  Stack          │  Stack               │
└─────────────┬──────────────────────┬──────────────────┬────────┘
              │                      │                  │
              ▼                      ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ROBOT PLATFORM                                │
├─────────────────────────────────────────────────────────────────┤
│  Humanoid Robot   │  Sensors        │  Actuators           │
│  (Isaac Sim)      │  (Cameras,      │  (Motors, Grippers)  │
│                   │   LiDAR, IMU)   │                      │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Environment Setup and Robot Configuration

```python
# capstone_project/launch/humanoid_robot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    world = DeclareLaunchArgument(
        'world',
        default_value='autonomous_house',
        description='Choose one of the world files from `/worlds`'
    )

    # Launch Gazebo with humanoid robot
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('capstone_project'),
                'worlds',
                LaunchConfiguration('world') + '.world'
            ])
        }.items()
    )

    # Launch humanoid robot
    robot_spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'humanoid_robot',
            '-file', PathJoinSubstitution([
                FindPackageShare('capstone_project'),
                'models',
                'humanoid_robot.urdf'
            ]),
            '-x', '0', '-y', '0', '-z', '1'
        ],
        output='screen'
    )

    # Launch robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': PathJoinSubstitution([
                FindPackageShare('capstone_project'),
                'urdf',
                'humanoid_robot.urdf'
            ])
        }]
    )

    return LaunchDescription([
        world,
        gazebo,
        robot_spawn,
        robot_state_publisher
    ])
```

### Phase 2: Perception System Integration

```python
# capstone_project/src/perception_system.py
#!/usr/bin/env python3

"""
Perception system for the autonomous humanoid robot.
Integrates Isaac ROS perception nodes with custom processing.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, LaserScan
from vision_msgs.msg import Detection2DArray, Detection2D
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.spatial import distance
import tf2_ros
from tf2_ros import TransformListener
from tf2_geometry_msgs import do_transform_point
import message_filters


class HumanoidPerceptionSystem(Node):
    """
    Integrated perception system for the humanoid robot.
    """

    def __init__(self):
        super().__init__('humanoid_perception_system')

        # CV Bridge for image processing
        self.cv_bridge = CvBridge()

        # TF2 for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # QoS for sensor data
        sensor_qos = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE
        )

        # Synchronized subscribers for RGB-D data
        rgb_sub = message_filters.Subscriber(self, Image, '/camera/rgb/image_raw', qos_profile=sensor_qos)
        depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw', qos_profile=sensor_qos)
        camera_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/rgb/camera_info', qos_profile=sensor_qos)

        # Synchronize RGB, depth, and camera info
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, camera_info_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.rgb_depth_callback)

        # Other sensor subscriptions
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/lidar/scan',
            self.lidar_callback,
            10
        )

        # Publishers
        self.object_detection_pub = self.create_publisher(
            Detection2DArray,
            '/perception/objects',
            10
        )

        self.navigation_map_pub = self.create_publisher(
            OccupancyGrid,
            '/perception/navigation_map',
            10
        )

        self.visualization_pub = self.create_publisher(
            MarkerArray,
            '/perception/visualization',
            10
        )

        # Object detection model
        self.object_detector = self.initialize_object_detector()

        # Perception parameters
        self.detection_confidence_threshold = 0.7
        self.min_object_size = 0.05  # meters
        self.max_detection_distance = 3.0  # meters

        # Internal state
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_intrinsics = None
        self.camera_info = None

        # Object tracking
        self.tracked_objects = {}
        self.next_object_id = 0

        self.get_logger().info('Humanoid Perception System initialized')

    def initialize_object_detector(self):
        """
        Initialize object detection model (using Isaac ROS or custom).
        """
        # In real implementation, this would load a trained object detection model
        # For now, we'll use a placeholder
        return None

    def rgb_depth_callback(self, rgb_msg, depth_msg, camera_info_msg):
        """
        Process synchronized RGB-D data.
        """
        try:
            # Convert ROS messages to OpenCV
            rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

            # Store camera info
            self.camera_info = camera_info_msg
            self.camera_intrinsics = np.array(camera_info_msg.k).reshape(3, 3)

            # Store images
            self.latest_rgb = rgb_image
            self.latest_depth = depth_image

            # Process perception pipeline
            self.perception_pipeline()

        except Exception as e:
            self.get_logger().error(f'Error processing RGB-D data: {e}')

    def lidar_callback(self, msg):
        """
        Process LiDAR data for navigation mapping.
        """
        # Process LiDAR data to create navigation map
        navigation_map = self.process_lidar_for_navigation(msg)

        if navigation_map is not None:
            self.navigation_map_pub.publish(navigation_map)

    def perception_pipeline(self):
        """
        Main perception pipeline processing.
        """
        if self.latest_rgb is None or self.latest_depth is None:
            return

        # Object detection
        detections = self.detect_objects(self.latest_rgb)

        # 3D object localization using depth
        localized_detections = self.localize_detections_in_3d(detections, self.latest_depth)

        # Track objects over time
        tracked_detections = self.track_objects(localized_detections)

        # Publish detections
        self.publish_detections(tracked_detections)

        # Update visualization
        self.update_visualization(tracked_detections)

    def detect_objects(self, image):
        """
        Detect objects in RGB image.
        """
        # In real implementation, use Isaac ROS object detection or custom model
        # For demonstration, use a simple color-based detection
        detections = []

        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define colors to detect (red, blue, green)
        colors = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255])
        }

        for color_name, (lower, upper) in colors.items():
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)

                    detection = Detection2D()
                    detection.header.frame_id = 'camera_link'
                    detection.header.stamp = self.get_clock().now().to_msg()

                    # Bounding box
                    detection.bbox.center.x = x + w/2
                    detection.bbox.center.y = y + h/2
                    detection.bbox.size_x = w
                    detection.bbox.size_y = h

                    # Classification
                    class_hypothesis = ObjectHypothesisWithPose()
                    class_hypothesis.hypothesis.class_id = color_name
                    class_hypothesis.hypothesis.score = 0.8
                    detection.results.append(class_hypothesis)

                    detections.append(detection)

        return detections

    def localize_detections_in_3d(self, detections, depth_image):
        """
        Localize 2D detections in 3D space using depth information.
        """
        localized_detections = []

        for detection in detections:
            # Get center of bounding box
            center_x = int(detection.bbox.center.x)
            center_y = int(detection.bbox.center.y)

            # Get depth at center (average in a small region for robustness)
            depth_region = depth_image[
                max(0, center_y-5):min(depth_image.shape[0], center_y+5),
                max(0, center_x-5):min(depth_image.shape[1], center_x+5)
            ]

            valid_depths = depth_region[np.isfinite(depth_region)]
            if len(valid_depths) > 0:
                avg_depth = np.mean(valid_depths)

                # Convert pixel coordinates to 3D world coordinates
                if self.camera_intrinsics is not None:
                    x_3d = (center_x - self.camera_intrinsics[0, 2]) * avg_depth / self.camera_intrinsics[0, 0]
                    y_3d = (center_y - self.camera_intrinsics[1, 2]) * avg_depth / self.camera_intrinsics[1, 1]
                    z_3d = avg_depth

                    # Transform from camera frame to robot base frame
                    try:
                        transform = self.tf_buffer.lookup_transform(
                            'base_link', 'camera_link',
                            rclpy.time.Time(seconds=0),
                            timeout_sec=0.1
                        )

                        # Apply transformation
                        point_in_camera = Point(x=x_3d, y=y_3d, z=z_3d)
                        point_in_base = do_transform_point(point_in_camera, transform)

                        detection.position_3d = point_in_base
                        localized_detections.append(detection)

                    except tf2_ros.TransformException as ex:
                        self.get_logger().warn(f'Could not transform point: {ex}')
                        # Use camera frame coordinates as fallback
                        detection.position_3d = Point(x=x_3d, y=y_3d, z=z_3d)
                        localized_detections.append(detection)
            else:
                # No valid depth, skip this detection
                continue

        return localized_detections

    def track_objects(self, detections):
        """
        Track objects over time using simple association.
        """
        current_time = self.get_clock().now()

        for detection in detections:
            # Simple nearest neighbor association
            best_match = None
            min_distance = float('inf')

            for obj_id, obj_info in self.tracked_objects.items():
                # Calculate distance between current detection and existing object
                dist = distance.euclidean(
                    [detection.position_3d.x, detection.position_3d.y, detection.position_3d.z],
                    [obj_info['position'].x, obj_info['position'].y, obj_info['position'].z]
                )

                if dist < min_distance and dist < 0.5:  # 50cm threshold
                    min_distance = dist
                    best_match = obj_id

            if best_match is not None:
                # Update existing track
                self.tracked_objects[best_match]['position'] = detection.position_3d
                self.tracked_objects[best_match]['last_seen'] = current_time
                self.tracked_objects[best_match]['detection'] = detection
            else:
                # Create new track
                new_id = self.next_object_id
                self.tracked_objects[new_id] = {
                    'id': new_id,
                    'position': detection.position_3d,
                    'detection': detection,
                    'last_seen': current_time,
                    'first_seen': current_time,
                    'class': detection.results[0].hypothesis.class_id if detection.results else 'unknown'
                }
                self.next_object_id += 1

        # Remove old tracks (not seen for more than 5 seconds)
        current_time = self.get_clock().now()
        ids_to_remove = []
        for obj_id, obj_info in self.tracked_objects.items():
            time_since_seen = (current_time - obj_info['last_seen']).nanoseconds / 1e9
            if time_since_seen > 5.0:
                ids_to_remove.append(obj_id)

        for obj_id in ids_to_remove:
            del self.tracked_objects[obj_id]

        # Return tracked detections
        return [obj_info['detection'] for obj_info in self.tracked_objects.values()]

    def publish_detections(self, detections):
        """
        Publish object detections.
        """
        detection_array = Detection2DArray()
        detection_array.header.frame_id = 'base_link'
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.detections = detections

        self.object_detection_pub.publish(detection_array)

    def update_visualization(self, detections):
        """
        Update visualization markers for detected objects.
        """
        marker_array = MarkerArray()

        for i, detection in enumerate(detections):
            # Create marker for each detection
            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'detected_objects'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Position from 3D localization
            marker.pose.position = detection.position_3d
            marker.pose.orientation.w = 1.0

            # Scale based on object size
            marker.scale.x = 0.1  # 10cm diameter
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            # Color based on class
            class_name = detection.results[0].hypothesis.class_id if detection.results else 'unknown'
            if class_name == 'red':
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif class_name == 'blue':
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            elif class_name == 'green':
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0  # Yellow for unknown

            marker.color.a = 0.8

            # Add label
            label_marker = Marker()
            label_marker.header = marker.header
            label_marker.ns = 'object_labels'
            label_marker.id = i + 1000  # Separate ID space
            label_marker.type = Marker.TEXT_VIEW_FACING
            label_marker.action = Marker.ADD
            label_marker.pose.position = detection.position_3d
            label_marker.pose.position.z += 0.2  # Raise label above object
            label_marker.pose.orientation.w = 1.0
            label_marker.text = f"{class_name}"
            label_marker.scale.z = 0.1
            label_marker.color.r = 1.0
            label_marker.color.g = 1.0
            label_marker.color.b = 1.0
            label_marker.color.a = 1.0

            marker_array.markers.append(marker)
            marker_array.markers.append(label_marker)

        self.visualization_pub.publish(marker_array)

    def process_lidar_for_navigation(self, lidar_msg):
        """
        Process LiDAR data to create navigation map.
        """
        # This would create an occupancy grid for navigation
        # For simplicity, we'll create a placeholder
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header.frame_id = 'map'
        occupancy_grid.header.stamp = lidar_msg.header.stamp

        # Set map parameters (simplified)
        occupancy_grid.info.resolution = 0.05  # 5cm resolution
        occupancy_grid.info.width = 400  # 20m x 20m map
        occupancy_grid.info.height = 400
        occupancy_grid.info.origin.position.x = -10.0
        occupancy_grid.info.origin.position.y = -10.0

        # Create empty map initially
        occupancy_grid.data = [-1] * (occupancy_grid.info.width * occupancy_grid.info.height)  # Unknown

        return occupancy_grid


class HumanoidNavigationSystem(Node):
    """
    Navigation system for the humanoid robot.
    """

    def __init__(self):
        super().__init__('humanoid_navigation_system')

        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Internal state
        self.current_pose = None
        self.current_twist = None
        self.current_orientation = None

        # Navigation parameters
        self.linear_speed = 0.3  # m/s
        self.angular_speed = 0.5  # rad/s
        self.arrival_threshold = 0.3  # meters
        self.rotation_threshold = 0.1  # radians

        self.get_logger().info('Humanoid Navigation System initialized')

    def navigate_to_pose(self, target_pose):
        """
        Navigate to target pose using navigation2.
        """
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation action server not available')
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.navigation_done_callback)

        return True

    def navigation_done_callback(self, future):
        """
        Callback when navigation goal is completed.
        """
        goal_handle = future.result()

        if goal_handle.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Navigation succeeded')
        elif goal_handle.status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info('Navigation was canceled')
        elif goal_handle.status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error('Navigation failed')
        else:
            self.get_logger().info(f'Navigation finished with status: {goal_handle.status}')

    def odom_callback(self, msg):
        """
        Update robot pose from odometry.
        """
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def imucallback(self, msg):
        """
        Update robot orientation from IMU.
        """
        self.current_orientation = msg.orientation


class HumanoidManipulationSystem(Node):
    """
    Manipulation system for the humanoid robot.
    """

    def __init__(self):
        super().__init__('humanoid_manipulation_system')

        # Publishers for arm control
        self.left_arm_pub = self.create_publisher(JointTrajectory, '/left_arm_controller/joint_trajectory', 10)
        self.right_arm_pub = self.create_publisher(JointTrajectory, '/right_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(Float64MultiArray, '/gripper_controller/commands', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Internal state
        self.current_joint_positions = {}
        self.left_arm_joints = ['left_shoulder_pan', 'left_shoulder_lift', 'left_elbow_flex',
                               'left_wrist_flex', 'left_wrist_roll', 'left_forearm_roll', 'left_gripper']
        self.right_arm_joints = ['right_shoulder_pan', 'right_shoulder_lift', 'right_elbow_flex',
                                'right_wrist_flex', 'right_wrist_roll', 'right_forearm_roll', 'right_gripper']

        self.get_logger().info('Humanoid Manipulation System initialized')

    def joint_state_callback(self, msg):
        """
        Update current joint positions.
        """
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

    def move_arm_to_pose(self, arm_side, target_pose, duration=5.0):
        """
        Move specified arm to target pose.
        """
        if arm_side not in ['left', 'right']:
            self.get_logger().error(f'Invalid arm side: {arm_side}')
            return False

        # Calculate joint trajectory for target pose
        # This would use inverse kinematics in real implementation
        joint_trajectory = self.calculate_arm_trajectory(arm_side, target_pose)

        # Publish trajectory
        if arm_side == 'left':
            self.left_arm_pub.publish(joint_trajectory)
        else:
            self.right_arm_pub.publish(joint_trajectory)

        return True

    def calculate_arm_trajectory(self, arm_side, target_pose):
        """
        Calculate joint trajectory for arm movement (simplified).
        """
        # This would use inverse kinematics in real implementation
        # For now, return a simple trajectory
        trajectory = JointTrajectory()
        trajectory.joint_names = self.left_arm_joints if arm_side == 'left' else self.right_arm_joints

        # Create trajectory point
        point = JointTrajectoryPoint()
        # In real implementation, calculate actual joint positions using IK
        point.positions = [0.0] * len(trajectory.joint_names)  # Placeholder
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1) * 1e9)

        trajectory.points = [point]
        return trajectory

    def grasp_object(self, arm_side):
        """
        Close gripper to grasp object.
        """
        commands = Float64MultiArray()
        if arm_side == 'left':
            commands.data = [0.0]  # Close left gripper
        else:
            commands.data = [0.0]  # Close right gripper

        self.gripper_pub.publish(commands)
        self.get_logger().info(f'{arm_side.capitalize()} gripper closed')

    def release_object(self, arm_side):
        """
        Open gripper to release object.
        """
        commands = Float64MultiArray()
        if arm_side == 'left':
            commands.data = [1.0]  # Open left gripper
        else:
            commands.data = [1.0]  # Open right gripper

        self.gripper_pub.publish(commands)
        self.get_logger().info(f'{arm_side.capitalize()} gripper opened')
```

### Phase 3: Cognitive Planning and Natural Language Integration

```python
# capstone_project/src/cognitive_planner.py
#!/usr/bin/env python3

"""
Cognitive planning system for the autonomous humanoid robot.
Integrates LLM-based planning with multi-modal interaction.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time
import asyncio
import json
import openai
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    INTERACTION = "interaction"


@dataclass
class Task:
    """
    Data structure for tasks in the planning system.
    """
    id: str
    type: TaskType
    description: str
    parameters: Dict
    priority: int = 0
    status: str = "pending"  # pending, executing, completed, failed
    dependencies: List[str] = None
    created_at: float = 0.0


class AutonomousHumanoidPlanner(Node):
    """
    Cognitive planning system for the humanoid robot.
    """

    def __init__(self):
        super().__init__('autonomous_humanoid_planner')

        # Initialize LLM client
        self.llm_client = self.initialize_llm_client()

        # Publishers
        self.task_pub = self.create_publisher(String, '/planning/tasks', 10)
        self.status_pub = self.create_publisher(String, '/planning/status', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String,
            '/natural_language_command',
            self.command_callback,
            10
        )

        self.perception_sub = self.create_subscription(
            String,
            '/perception/events',
            self.perception_callback,
            10
        )

        # Internal state
        self.current_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.robot_capabilities = self.define_robot_capabilities()
        self.location_map = self.define_location_map()

        # Task execution
        self.current_task = None
        self.task_executor = TaskExecutor(self)

        self.get_logger().info('Autonomous Humanoid Planner initialized')

    def initialize_llm_client(self):
        """
        Initialize LLM client for cognitive planning.
        """
        # In practice, set your API key securely
        # openai.api_key = os.getenv("OPENAI_API_KEY")
        return openai

    def define_robot_capabilities(self):
        """
        Define robot's capabilities for planning.
        """
        return {
            'locomotion': {
                'abilities': ['navigate', 'move_base', 'turn'],
                'constraints': ['max_speed: 0.5 m/s', 'indoor_only']
            },
            'manipulation': {
                'abilities': ['grasp', 'release', 'carry', 'place'],
                'constraints': ['max_payload: 2.0 kg', 'reachable_distance: 1.5m']
            },
            'perception': {
                'abilities': ['detect_objects', 'recognize_people', 'measure_distances'],
                'constraints': ['fov: 60 degrees', 'range: 0.1-3.0m']
            },
            'communication': {
                'abilities': ['speak', 'listen', 'gesture'],
                'constraints': ['language: English']
            }
        }

    def define_location_map(self):
        """
        Define known locations in the environment.
        """
        return {
            'kitchen': {'x': -2.0, 'y': 1.0, 'description': 'Kitchen area with appliances'},
            'living_room': {'x': 1.0, 'y': -1.0, 'description': 'Main living area'},
            'bedroom': {'x': 2.0, 'y': 2.0, 'description': 'Sleeping area'},
            'office': {'x': -1.0, 'y': -2.0, 'description': 'Work area'},
            'entrance': {'x': 0.0, 'y': 0.0, 'description': 'Main entrance'},
            'dining_room': {'x': -1.5, 'y': 0.5, 'description': 'Dining area'}
        }

    def command_callback(self, msg):
        """
        Process natural language command.
        """
        command_text = msg.data
        self.get_logger().info(f'Received command: {command_text}')

        # Plan and execute command
        asyncio.create_task(self.process_command_async(command_text))

    def perception_callback(self, msg):
        """
        Process perception events for context-aware planning.
        """
        try:
            perception_data = json.loads(msg.data)
            self.update_context_with_perception(perception_data)
        except json.JSONDecodeError:
            self.get_logger().error('Could not parse perception data')

    def update_context_with_perception(self, perception_data):
        """
        Update planning context with perception information.
        """
        # Update known objects and locations based on perception
        if 'objects' in perception_data:
            for obj in perception_data['objects']:
                # Add object to context if not already known
                pass

    async def process_command_async(self, command_text):
        """
        Process command asynchronously using LLM.
        """
        try:
            # Generate plan using LLM
            plan = await self.generate_plan_with_llm(command_text)

            # Execute plan
            await self.execute_plan_async(plan)

        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')
            self.publish_status(f'Error: {str(e)}')

    async def generate_plan_with_llm(self, command_text):
        """
        Generate plan using LLM based on command.
        """
        system_prompt = f"""
        You are a cognitive planning assistant for a humanoid robot. The robot has these capabilities:

        {json.dumps(self.robot_capabilities, indent=2)}

        Known locations in the environment:
        {json.dumps(self.location_map, indent=2)}

        Your task is to break down the following command into executable steps:
        "{command_text}"

        Respond with a JSON array of tasks, where each task has:
        - 'type': task type (navigation, manipulation, perception, interaction)
        - 'action': specific action to take
        - 'parameters': any required parameters
        - 'description': human-readable description
        - 'dependencies': list of task IDs that must be completed first

        Tasks should be atomic and executable by the robot.
        """

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm_client.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": command_text}
                    ],
                    temperature=0.1
                )
            )

            plan_text = response.choices[0].message.content

            # Parse plan from response
            plan = self.parse_plan_from_response(plan_text)

            # Validate and prioritize plan
            validated_plan = self.validate_and_prioritize_plan(plan)

            self.get_logger().info(f'Generated plan: {validated_plan}')

            return validated_plan

        except Exception as e:
            self.get_logger().error(f'Error generating plan: {e}')
            return []

    def parse_plan_from_response(self, response_text):
        """
        Parse plan from LLM response.
        """
        try:
            # Look for JSON in the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                plan = json.loads(json_str)
                return plan
            else:
                # If no JSON found, create simple plan based on command
                return self.create_simple_plan_from_command(response_text)
        except json.JSONDecodeError:
            self.get_logger().error('Could not parse LLM response as JSON')
            return []

    def create_simple_plan_from_command(self, command_text):
        """
        Create simple plan if LLM response couldn't be parsed.
        """
        plan = []

        if 'go to' in command_text.lower() or 'navigate to' in command_text.lower():
            # Extract location
            for loc_name in self.location_map.keys():
                if loc_name in command_text.lower():
                    plan.append({
                        'type': 'navigation',
                        'action': 'go_to_location',
                        'parameters': {'location': loc_name},
                        'description': f'Navigate to {loc_name}',
                        'dependencies': []
                    })
                    break

        elif 'pick up' in command_text.lower() or 'grasp' in command_text.lower():
            plan.append({
                'type': 'manipulation',
                'action': 'grasp_object',
                'parameters': {'object': 'unknown'},  # Would be extracted from command
                'description': 'Grasp specified object',
                'dependencies': [{'type': 'navigation', 'action': 'go_to_location'}]  # Navigate first
            })

        elif 'find' in command_text.lower() or 'look for' in command_text.lower():
            plan.append({
                'type': 'perception',
                'action': 'detect_objects',
                'parameters': {'object_type': 'unknown'},
                'description': 'Detect specified object',
                'dependencies': []
            })

        return plan

    def validate_and_prioritize_plan(self, plan):
        """
        Validate and prioritize the generated plan.
        """
        validated_plan = []

        for i, task_data in enumerate(plan):
            try:
                # Validate task data
                task_type = task_data.get('type')
                action = task_data.get('action')
                parameters = task_data.get('parameters', {})
                description = task_data.get('description', f'Task {i}')
                dependencies = task_data.get('dependencies', [])

                if not task_type or not action:
                    continue  # Skip invalid tasks

                # Create task object
                task = Task(
                    id=f'task_{i}',
                    type=TaskType(task_type),
                    description=description,
                    parameters=parameters,
                    dependencies=dependencies,
                    created_at=time.time()
                )

                # Validate against robot capabilities
                if self.is_task_feasible(task):
                    validated_plan.append(task)

            except (KeyError, ValueError) as e:
                self.get_logger().warn(f'Invalid task data: {task_data}, error: {e}')
                continue

        return validated_plan

    def is_task_feasible(self, task: Task) -> bool:
        """
        Check if a task is feasible given robot capabilities.
        """
        # Check if action is supported by robot
        for capability_category in self.robot_capabilities.values():
            if task.description in capability_category['abilities']:
                return True

        # For more complex checks, examine action and parameters
        if task.type == TaskType.NAVIGATION:
            # Check navigation constraints
            return True
        elif task.type == TaskType.MANIPULATION:
            # Check manipulation constraints
            return True
        elif task.type == TaskType.PERCEPTION:
            # Check perception constraints
            return True
        elif task.type == TaskType.INTERACTION:
            # Check interaction constraints
            return True

        return False

    async def execute_plan_async(self, plan):
        """
        Execute the plan asynchronously.
        """
        if not plan:
            self.get_logger().warn('No plan to execute')
            return

        self.get_logger().info(f'Executing plan with {len(plan)} tasks')

        # Add tasks to queue
        for task in plan:
            self.current_tasks.append(task)

        # Execute tasks in order respecting dependencies
        await self.execute_task_queue()

    async def execute_task_queue(self):
        """
        Execute tasks in the queue respecting dependencies.
        """
        completed_ids = set()

        while self.current_tasks:
            # Find tasks whose dependencies are satisfied
            ready_tasks = []
            for task in self.current_tasks:
                deps_satisfied = True
                for dep in task.dependencies:
                    if isinstance(dep, dict) and 'id' in dep:
                        if dep['id'] not in completed_ids:
                            deps_satisfied = False
                            break
                    elif isinstance(dep, str):
                        if dep not in completed_ids:
                            deps_satisfied = False
                            break

                if deps_satisfied:
                    ready_tasks.append(task)

            if not ready_tasks:
                self.get_logger().error('Deadlock detected in task dependencies')
                break

            # Execute ready tasks (in parallel or sequentially based on type)
            for task in ready_tasks:
                self.current_tasks.remove(task)
                success = await self.execute_single_task(task)

                if success:
                    completed_ids.add(task.id)
                    self.completed_tasks.append(task)
                    self.publish_status(f'Task completed: {task.description}')
                else:
                    self.failed_tasks.append(task)
                    self.publish_status(f'Task failed: {task.description}')
                    # Continue with other tasks

    async def execute_single_task(self, task: Task) -> bool:
        """
        Execute a single task.
        """
        self.get_logger().info(f'Executing task: {task.description}')

        # Update task status
        task.status = 'executing'
        self.publish_task_status(task)

        try:
            if task.type == TaskType.NAVIGATION:
                return await self.execute_navigation_task(task)
            elif task.type == TaskType.MANIPULATION:
                return await self.execute_manipulation_task(task)
            elif task.type == TaskType.PERCEPTION:
                return await self.execute_perception_task(task)
            elif task.type == TaskType.INTERACTION:
                return await self.execute_interaction_task(task)
            else:
                self.get_logger().error(f'Unknown task type: {task.type}')
                return False

        except Exception as e:
            self.get_logger().error(f'Error executing task {task.id}: {e}')
            return False

    async def execute_navigation_task(self, task: Task) -> bool:
        """
        Execute navigation task.
        """
        location_name = task.parameters.get('location')

        if not location_name:
            self.get_logger().error('No location specified for navigation task')
            return False

        # Get coordinates for location
        if location_name not in self.location_map:
            self.get_logger().error(f'Unknown location: {location_name}')
            return False

        location_data = self.location_map[location_name]
        target_pose = PoseStamped()
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.header.frame_id = 'map'
        target_pose.pose.position.x = location_data['x']
        target_pose.pose.position.y = location_data['y']
        target_pose.pose.position.z = 0.0

        # Call navigation system
        # This would integrate with the navigation system created earlier
        self.get_logger().info(f'Navigating to {location_name} at ({location_data["x"]}, {location_data["y"]})')

        # Simulate navigation execution
        await asyncio.sleep(2.0)  # Simulated navigation time

        return True

    async def execute_manipulation_task(self, task: Task) -> bool:
        """
        Execute manipulation task.
        """
        action = task.parameters.get('action')
        object_name = task.parameters.get('object')

        if not action:
            self.get_logger().error('No action specified for manipulation task')
            return False

        self.get_logger().info(f'Performing manipulation: {action} on {object_name}')

        # Call manipulation system
        # This would integrate with the manipulation system created earlier
        if action == 'grasp_object':
            # Simulate grasping
            await asyncio.sleep(1.0)
        elif action == 'release_object':
            # Simulate releasing
            await asyncio.sleep(1.0)

        return True

    async def execute_perception_task(self, task: Task) -> bool:
        """
        Execute perception task.
        """
        object_type = task.parameters.get('object_type')

        self.get_logger().info(f'Detecting objects of type: {object_type}')

        # Call perception system
        # This would integrate with the perception system created earlier

        # Simulate perception execution
        await asyncio.sleep(1.0)

        return True

    async def execute_interaction_task(self, task: Task) -> bool:
        """
        Execute interaction task.
        """
        message = task.parameters.get('message', '')

        self.get_logger().info(f'Interacting: {message}')

        # Call speech system
        # This would integrate with speech system

        # Simulate interaction execution
        await asyncio.sleep(0.5)

        return True

    def publish_task_status(self, task: Task):
        """
        Publish task status update.
        """
        status_msg = String()
        status_msg.data = f'TASK_STATUS: {task.id} - {task.status} - {task.description}'
        self.status_pub.publish(status_msg)

    def publish_status(self, status_text: str):
        """
        Publish general status update.
        """
        status_msg = String()
        status_msg.data = status_text
        self.status_pub.publish(status_msg)


class TaskExecutor:
    """
    Component responsible for executing individual tasks.
    """

    def __init__(self, planner_node):
        self.planner_node = planner_node
        self.execution_history = []

    async def execute_task(self, task: Task) -> bool:
        """
        Execute a task and return success status.
        """
        start_time = time.time()

        success = await self.planner_node.execute_single_task(task)

        execution_time = time.time() - start_time

        # Record execution
        self.execution_history.append({
            'task_id': task.id,
            'success': success,
            'execution_time': execution_time,
            'timestamp': time.time()
        })

        return success
```

### Phase 4: Integration and Testing

```python
# capstone_project/src/main_system.py
#!/usr/bin/env python3

"""
Main system integration for the autonomous humanoid robot.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import asyncio
import threading
import time


class AutonomousHumanoidSystem(Node):
    """
    Main system node that integrates all components.
    """

    def __init__(self):
        super().__init__('autonomous_humanoid_system')

        # Initialize subsystems
        self.perception_system = HumanoidPerceptionSystem()
        self.navigation_system = HumanoidNavigationSystem()
        self.manipulation_system = HumanoidManipulationSystem()
        self.cognitive_planner = AutonomousHumanoidPlanner()

        # Publishers
        self.system_status_pub = self.create_publisher(String, '/system/status', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String,
            '/user/command',
            self.user_command_callback,
            10
        )

        # System state
        self.system_state = 'idle'  # idle, active, paused, error
        self.active_components = []

        # Timers
        self.status_timer = self.create_timer(1.0, self.publish_system_status)

        self.get_logger().info('Autonomous Humanoid System initialized')

    def user_command_callback(self, msg):
        """
        Handle user commands that may involve multiple subsystems.
        """
        command = msg.data
        self.get_logger().info(f'Received user command: {command}')

        # Route command to appropriate subsystem
        if any(word in command.lower() for word in ['go', 'navigate', 'move', 'walk']):
            # Route to cognitive planner for high-level navigation
            cmd_msg = String()
            cmd_msg.data = command
            self.cognitive_planner.command_pub.publish(cmd_msg)
        elif any(word in command.lower() for word in ['pick', 'grasp', 'take', 'lift']):
            # Route to cognitive planner for manipulation
            cmd_msg = String()
            cmd_msg.data = command
            self.cognitive_planner.command_pub.publish(cmd_msg)
        elif any(word in command.lower() for word in ['find', 'look', 'detect', 'see']):
            # Route to cognitive planner for perception
            cmd_msg = String()
            cmd_msg.data = command
            self.cognitive_planner.command_pub.publish(cmd_msg)
        else:
            # Default to cognitive planner for complex commands
            cmd_msg = String()
            cmd_msg.data = command
            self.cognitive_planner.command_pub.publish(cmd_msg)

    def publish_system_status(self):
        """
        Publish overall system status.
        """
        status_msg = String()
        status_msg.data = f'SYSTEM_STATUS: {self.system_state} - Components: {len(self.active_components)}'
        self.system_status_pub.publish(status_msg)

    def start_system(self):
        """
        Start all subsystems.
        """
        self.system_state = 'active'
        self.get_logger().info('Starting autonomous humanoid system...')

        # Start each subsystem in separate threads
        self.perception_thread = threading.Thread(target=self.run_perception, daemon=True)
        self.navigation_thread = threading.Thread(target=self.run_navigation, daemon=True)
        self.manipulation_thread = threading.Thread(target=self.run_manipulation, daemon=True)
        self.planning_thread = threading.Thread(target=self.run_planning, daemon=True)

        self.perception_thread.start()
        self.navigation_thread.start()
        self.manipulation_thread.start()
        self.planning_thread.start()

        self.active_components = ['perception', 'navigation', 'manipulation', 'planning']
        self.get_logger().info('All subsystems started')

    def run_perception(self):
        """
        Run perception system continuously.
        """
        # In real implementation, this would run the perception system
        pass

    def run_navigation(self):
        """
        Run navigation system continuously.
        """
        # In real implementation, this would run the navigation system
        pass

    def run_manipulation(self):
        """
        Run manipulation system continuously.
        """
        # In real implementation, this would run the manipulation system
        pass

    def run_planning(self):
        """
        Run planning system continuously.
        """
        # In real implementation, this would run the planning system
        pass

    def stop_system(self):
        """
        Stop all subsystems.
        """
        self.system_state = 'idle'
        self.get_logger().info('Stopping autonomous humanoid system...')

        # Stop all threads
        # In real implementation, set flags to stop each subsystem
        pass


def main(args=None):
    """
    Main function to run the autonomous humanoid system.
    """
    rclpy.init(args=args)

    system = AutonomousHumanoidSystem()

    try:
        system.start_system()
        rclpy.spin(system)
    except KeyboardInterrupt:
        system.get_logger().info('Shutting down autonomous humanoid system...')
    finally:
        system.stop_system()
        system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Testing and Validation

### Test Scenarios

1. **Basic Navigation Test**:
   - Command: "Go to the kitchen"
   - Expected: Robot navigates to kitchen location

2. **Object Manipulation Test**:
   - Command: "Pick up the red ball from the living room"
   - Expected: Robot navigates to living room, detects red ball, grasps it

3. **Multi-step Task Test**:
   - Command: "Go to the kitchen, find a cup, bring it to me"
   - Expected: Robot performs sequence of navigation, detection, manipulation

4. **Social Interaction Test**:
   - Command: "Say hello to everyone in the room"
   - Expected: Robot detects people and greets them

### Performance Metrics

- **Success Rate**: Percentage of tasks completed successfully
- **Response Time**: Time from command to action initiation
- **Accuracy**: Precision of navigation and manipulation
- **Robustness**: Performance under various conditions
- **User Satisfaction**: Subjective measure of naturalness

## Conclusion

The autonomous humanoid robot capstone project demonstrates the integration of all the concepts covered in this book. It showcases how ROS 2, simulation environments, AI perception and planning, and multi-modal interaction can work together to create an intelligent, autonomous robot system capable of understanding natural language commands and performing complex tasks in real-world environments.

The system architecture is modular and extensible, allowing for future enhancements and adaptations. Through this project, you have gained hands-on experience with the entire pipeline of humanoid robotics development, from low-level control to high-level cognitive planning.
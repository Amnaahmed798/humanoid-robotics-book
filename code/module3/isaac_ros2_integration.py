#!/usr/bin/env python3

"""
Isaac ROS Integration Node

This example demonstrates how to integrate NVIDIA Isaac components with ROS 2,
including perception, navigation, and manipulation capabilities using Isaac ROS packages.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, LaserScan, Imu
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header, Float32
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time
import numpy as np
from cv_bridge import CvBridge
import tf_transformations
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import tf2_ros
import tf2_geometry_msgs
from message_filters import ApproximateTimeSynchronizer, Subscriber
import message_filters
import threading
import time


class IsaacROSPerceptionNode(Node):
    """
    Isaac ROS perception node integrating multiple sensors and perception algorithms.
    """

    def __init__(self):
        super().__init__('isaac_ros_perception_node')

        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # CV Bridge for image processing
        self.bridge = CvBridge()

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribers for Isaac Sim sensors
        self.rgb_sub = self.create_subscription(
            Image,
            '/rgb_camera/image',
            self.rgb_callback,
            sensor_qos
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/depth_camera/image',
            self.depth_callback,
            sensor_qos
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/rgb_camera/camera_info',
            self.camera_info_callback,
            sensor_qos
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/lidar/points',
            self.pointcloud_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Publishers for processed data
        self.object_detection_pub = self.create_publisher(
            MarkerArray,
            '/isaac_perception/objects',
            10
        )

        self.obstacle_pub = self.create_publisher(
            MarkerArray,
            '/isaac_perception/obstacles',
            10
        )

        self.free_space_pub = self.create_publisher(
            MarkerArray,
            '/isaac_perception/free_space',
            10
        )

        # Internal state
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None
        self.latest_imu = None
        self.latest_pointcloud = None
        self.camera_intrinsics = None

        # Perception parameters
        self.detection_threshold = 0.5
        self.obstacle_distance_threshold = 1.0  # meters
        self.min_obstacle_height = 0.1  # meters (ignore ground-level obstacles)

        # Initialize
        self.get_logger().info('Isaac ROS Perception Node initialized')

    def rgb_callback(self, msg):
        """
        Process RGB camera data from Isaac Sim.
        """
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().debug(f'Received RGB image: {msg.width}x{msg.height}')
        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')

    def depth_callback(self, msg):
        """
        Process depth camera data from Isaac Sim.
        """
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.get_logger().debug(f'Received depth image: {msg.width}x{msg.height}')
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def camera_info_callback(self, msg):
        """
        Process camera info from Isaac Sim.
        """
        self.camera_info = msg
        self.camera_intrinsics = np.array(msg.k).reshape(3, 3)
        self.get_logger().debug('Received camera info')

    def pointcloud_callback(self, msg):
        """
        Process point cloud data from Isaac Sim LiDAR.
        """
        self.latest_pointcloud = msg
        self.get_logger().debug('Received point cloud')

    def imu_callback(self, msg):
        """
        Process IMU data from Isaac Sim.
        """
        self.latest_imu = msg
        # Extract orientation for robot pose
        self.current_orientation = msg.orientation
        self.get_logger().debug('Received IMU data')

    def process_perception_data(self):
        """
        Process all sensor data for perception tasks.
        """
        if self.latest_rgb is None or self.latest_depth is None:
            return

        # Perform object detection on RGB image
        detected_objects = self.detect_objects(self.latest_rgb)

        # Analyze depth data for obstacles
        obstacles = self.find_obstacles(self.latest_depth)

        # Process point cloud for 3D obstacles
        if self.latest_pointcloud:
            pointcloud_obstacles = self.process_pointcloud(self.latest_pointcloud)
            obstacles.extend(pointcloud_obstacles)

        # Publish results
        self.publish_detected_objects(detected_objects)
        self.publish_obstacles(obstacles)

    def detect_objects(self, image):
        """
        Perform object detection on RGB image.
        This is a simplified implementation - in practice, use Isaac ROS perception packages.
        """
        # This would use Isaac ROS detection nodes in real implementation
        # For demonstration, we'll simulate object detection
        detected_objects = []

        # Simulate detecting a few objects (in real implementation, use actual detection)
        height, width = image.shape[:2]

        # Example: detect a "person" at center
        if np.random.random() > 0.7:  # 30% chance of detecting a person
            center_x, center_y = width // 2, height // 2
            bbox_width, bbox_height = width // 4, height // 3

            obj = {
                'class': 'person',
                'confidence': 0.85,
                'bbox': [center_x - bbox_width//2, center_y - bbox_height//2,
                         center_x + bbox_width//2, center_y + bbox_height//2],
                'center': (center_x, center_y)
            }
            detected_objects.append(obj)

        # Example: detect a "chair" at left
        if np.random.random() > 0.8:  # 20% chance of detecting a chair
            center_x, center_y = width // 4, height * 2 // 3
            bbox_width, bbox_height = width // 6, height // 4

            obj = {
                'class': 'chair',
                'confidence': 0.78,
                'bbox': [center_x - bbox_width//2, center_y - bbox_height//2,
                         center_x + bbox_width//2, center_y + bbox_height//2],
                'center': (center_x, center_y)
            }
            detected_objects.append(obj)

        return detected_objects

    def find_obstacles(self, depth_image):
        """
        Find obstacles using depth image.
        """
        obstacles = []

        # Find pixels with distance less than threshold
        valid_depths = depth_image[np.isfinite(depth_image)]
        if len(valid_depths) == 0:
            return obstacles

        # Create a simple obstacle map based on depth
        height, width = depth_image.shape
        grid_size = 20  # Size of each grid cell in pixels

        for y in range(0, height, grid_size):
            for x in range(0, width, grid_size):
                # Get average depth in this grid cell
                y_end = min(y + grid_size, height)
                x_end = min(x + grid_size, width)
                cell_depth = depth_image[y:y_end, x:x_end]

                if cell_depth.size > 0:
                    avg_depth = np.nanmean(cell_depth[np.isfinite(cell_depth)])

                    if np.isfinite(avg_depth) and avg_depth < self.obstacle_distance_threshold:
                        # Convert to 3D world coordinates (simplified)
                        if self.camera_intrinsics is not None:
                            # Calculate 3D point from pixel and depth
                            center_x, center_y = x + grid_size//2, y + grid_size//2
                            z = avg_depth  # Depth in meters
                            x_3d = (center_x - self.camera_intrinsics[0, 2]) * z / self.camera_intrinsics[0, 0]
                            y_3d = (center_y - self.camera_intrinsics[1, 2]) * z / self.camera_intrinsics[1, 1]

                            obstacle = {
                                'position': (x_3d, y_3d, z),
                                'depth': avg_depth,
                                'pixel_coords': (center_x, center_y)
                            }
                            obstacles.append(obstacle)

        return obstacles

    def process_pointcloud(self, pointcloud_msg):
        """
        Process point cloud data for obstacle detection.
        """
        # This would use Isaac ROS point cloud processing in real implementation
        obstacles = []

        # Simplified point cloud processing
        # In real implementation, use Isaac ROS point cloud packages

        # Extract points from message (simplified)
        # In practice, use sensor_msgs.point_cloud2.read_points() or similar

        return obstacles

    def publish_detected_objects(self, objects):
        """
        Publish detected objects as visualization markers.
        """
        marker_array = MarkerArray()

        for i, obj in enumerate(objects):
            if obj['confidence'] < self.detection_threshold:
                continue

            marker = Marker()
            marker.header.frame_id = 'camera_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'detected_objects'
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD

            # Position based on bounding box center
            center_x, center_y = obj['center']
            # Convert pixel to 3D if depth is available
            # For now, place at arbitrary distance
            marker.pose.position.x = (center_x - self.camera_intrinsics[0, 2]) * 2.0 / self.camera_intrinsics[0, 0] if self.camera_intrinsics is not None else 0.0
            marker.pose.position.y = (center_y - self.camera_intrinsics[1, 2]) * 2.0 / self.camera_intrinsics[1, 1] if self.camera_intrinsics is not None else 0.0
            marker.pose.position.z = 2.0  # 2 meters in front of camera

            marker.pose.orientation.w = 1.0
            marker.scale.z = 0.2  # Text size
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.text = f"{obj['class']}: {obj['confidence']:.2f}"

            marker_array.markers.append(marker)

            # Also add a bounding box
            bbox_marker = Marker()
            bbox_marker.header = marker.header
            bbox_marker.ns = 'object_bboxes'
            bbox_marker.id = i + 1000  # Separate ID space
            bbox_marker.type = Marker.LINE_STRIP
            bbox_marker.action = Marker.ADD
            bbox_marker.scale.x = 0.02
            bbox_marker.color.r = 1.0
            bbox_marker.color.a = 1.0

            # Define rectangle points
            x1, y1, x2, y2 = obj['bbox']
            z = marker.pose.position.z  # Same depth as text

            # Convert bbox corners to 3D points
            p1 = Point()
            p1.x = (x1 - self.camera_intrinsics[0, 2]) * z / self.camera_intrinsics[0, 0] if self.camera_intrinsics is not None else 0.0
            p1.y = (y1 - self.camera_intrinsics[1, 2]) * z / self.camera_intrinsics[1, 1] if self.camera_intrinsics is not None else 0.0
            p1.z = z

            p2 = Point()
            p2.x = (x2 - self.camera_intrinsics[0, 2]) * z / self.camera_intrinsics[0, 0] if self.camera_intrinsics is not None else 0.0
            p2.y = (y1 - self.camera_intrinsics[1, 2]) * z / self.camera_intrinsics[1, 1] if self.camera_intrinsics is not None else 0.0
            p2.z = z

            p3 = Point()
            p3.x = (x2 - self.camera_intrinsics[0, 2]) * z / self.camera_intrinsics[0, 0] if self.camera_intrinsics is not None else 0.0
            p3.y = (y2 - self.camera_intrinsics[1, 2]) * z / self.camera_intrinsics[1, 1] if self.camera_intrinsics is not None else 0.0
            p3.z = z

            p4 = Point()
            p4.x = (x1 - self.camera_intrinsics[0, 2]) * z / self.camera_intrinsics[0, 0] if self.camera_intrinsics is not None else 0.0
            p4.y = (y2 - self.camera_intrinsics[1, 2]) * z / self.camera_intrinsics[1, 1] if self.camera_intrinsics is not None else 0.0
            p4.z = z

            bbox_marker.points = [p1, p2, p3, p4, p1]  # Close the rectangle
            marker_array.markers.append(bbox_marker)

        self.object_detection_pub.publish(marker_array)

    def publish_obstacles(self, obstacles):
        """
        Publish obstacles as visualization markers.
        """
        marker_array = MarkerArray()

        for i, obstacle in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = 'base_link'  # Or appropriate frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'obstacles'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Position in 3D space
            marker.pose.position.x = obstacle['position'][0]
            marker.pose.position.y = obstacle['position'][1]
            marker.pose.position.z = obstacle['position'][2]

            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2  # Sphere size
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 1.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        self.obstacle_pub.publish(marker_array)


class IsaacROSNavigateNode(Node):
    """
    Isaac ROS navigation node with perception integration.
    """

    def __init__(self):
        super().__init__('isaac_ros_navigate_node')

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.perception_sub = self.create_subscription(
            MarkerArray,
            '/isaac_perception/obstacles',
            self.obstacle_callback,
            10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/goal',
            10
        )

        # Navigation parameters
        self.linear_speed = 0.3  # m/s
        self.angular_speed = 0.5  # rad/s
        self.safety_distance = 0.8  # meters
        self.arrival_threshold = 0.3  # meters

        # Robot state
        self.current_pose = None
        self.current_twist = None
        self.obstacles = []
        self.goal = None

        # Timer for navigation loop
        self.nav_timer = self.create_timer(0.1, self.navigation_loop)

        self.get_logger().info('Isaac ROS Navigation Node initialized')

    def odom_callback(self, msg):
        """
        Update robot pose from odometry.
        """
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def obstacle_callback(self, msg):
        """
        Update obstacle information from perception.
        """
        self.obstacles = []
        for marker in msg.markers:
            obstacle = {
                'position': (marker.pose.position.x, marker.pose.position.y, marker.pose.position.z),
                'size': (marker.scale.x, marker.scale.y, marker.scale.z)
            }
            self.obstacles.append(obstacle)

    def navigation_loop(self):
        """
        Main navigation control loop.
        """
        if self.current_pose is None or self.goal is None:
            return

        # Calculate distance and angle to goal
        dx = self.goal.pose.position.x - self.current_pose.position.x
        dy = self.goal.pose.position.y - self.current_pose.position.y
        distance_to_goal = np.sqrt(dx**2 + dy**2)

        # Check if goal reached
        if distance_to_goal < self.arrival_threshold:
            self.stop_robot()
            self.get_logger().info('Goal reached!')
            self.goal = None  # Clear goal
            return

        # Get robot orientation
        quat = self.current_pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        current_yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Calculate goal angle
        goal_angle = np.arctan2(dy, dx)

        # Check for obstacles in path
        obstacle_in_path = self.check_obstacle_in_path(dx, dy)

        # Create twist command
        cmd_vel = Twist()

        if obstacle_in_path:
            # Obstacle avoidance behavior
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = self.angular_speed  # Turn to avoid
            self.get_logger().info('Obstacle detected, executing avoidance')
        else:
            # Navigate toward goal
            angle_diff = self.normalize_angle(goal_angle - current_yaw)

            if abs(angle_diff) > 0.2:  # Need to turn
                cmd_vel.angular.z = self.angular_speed * np.sign(angle_diff)
                cmd_vel.linear.x = 0.0
            else:  # Move forward
                cmd_vel.linear.x = min(self.linear_speed, distance_to_goal)
                cmd_vel.angular.z = 0.0

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

    def check_obstacle_in_path(self, dx, dy):
        """
        Check if there are obstacles in the path to the goal.
        """
        # Calculate direction vector to goal
        path_direction = np.array([dx, dy])
        path_length = np.linalg.norm(path_direction)

        if path_length < 0.1:  # Very close to goal
            return False

        # Normalize direction
        path_direction = path_direction / path_length

        # Check obstacles within path corridor
        for obstacle in self.obstacles:
            obs_pos = np.array(obstacle['position'][:2])  # x, y only
            robot_pos = np.array([
                self.current_pose.position.x,
                self.current_pose.position.y
            ])

            # Vector from robot to obstacle
            robot_to_obs = obs_pos - robot_pos

            # Project obstacle onto path
            path_projection = np.dot(robot_to_obs, path_direction)

            # Check if obstacle is ahead and within safety corridor
            if 0 < path_projection < path_length:  # Obstacle is between robot and goal
                # Calculate perpendicular distance from path
                path_vector = path_direction * path_projection
                perpendicular_vector = robot_to_obs - path_vector
                perpendicular_distance = np.linalg.norm(perpendicular_vector)

                # Check if obstacle is within safety corridor
                if perpendicular_distance < self.safety_distance:
                    return True

        return False

    def normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi] range.
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def stop_robot(self):
        """
        Stop the robot.
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)


class IsaacROSManipulationNode(Node):
    """
    Isaac ROS manipulation node for controlling robot arms.
    """

    def __init__(self):
        super().__init__('isaac_ros_manipulation_node')

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Publishers
        self.joint_cmd_pub = self.create_publisher(
            JointState,
            '/joint_commands',
            10
        )

        self.ee_pose_pub = self.create_publisher(
            PoseStamped,
            '/end_effector_pose',
            10
        )

        # Manipulation parameters
        self.joint_names = []  # Will be populated from first joint state
        self.current_joint_positions = []
        self.target_joint_positions = []

        # Timer for manipulation control
        self.manip_timer = self.create_timer(0.05, self.manipulation_loop)

        self.get_logger().info('Isaac ROS Manipulation Node initialized')

    def joint_state_callback(self, msg):
        """
        Update joint state information.
        """
        if not self.joint_names:
            self.joint_names = list(msg.name)

        self.current_joint_positions = list(msg.position)

    def manipulation_loop(self):
        """
        Main manipulation control loop.
        """
        if not self.current_joint_positions:
            return

        # Example: Send a simple trajectory to move joints
        # In real implementation, this would follow a planned trajectory
        target_positions = self.generate_trajectory()

        # Create and publish joint command
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.name = self.joint_names
        cmd_msg.position = target_positions
        cmd_msg.velocity = [0.0] * len(target_positions)  # Zero velocity
        cmd_msg.effort = [0.0] * len(target_positions)    # Zero effort

        self.joint_cmd_pub.publish(cmd_msg)

        # Publish end-effector pose (calculated from forward kinematics)
        ee_pose = self.calculate_end_effector_pose(target_positions)
        if ee_pose:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'base_link'
            pose_msg.pose = ee_pose
            self.ee_pose_pub.publish(pose_msg)

    def generate_trajectory(self):
        """
        Generate a simple joint trajectory for demonstration.
        """
        if not self.current_joint_positions:
            return [0.0] * 7  # Default 7-DOF arm

        # Simple oscillating trajectory for demonstration
        current_time = self.get_clock().now().nanoseconds / 1e9
        amplitude = 0.5
        frequency = 0.5

        target_positions = []
        for i, current_pos in enumerate(self.current_joint_positions):
            # Different oscillation for each joint
            offset = i * 0.5  # Phase offset
            target_pos = current_pos + amplitude * np.sin(2 * np.pi * frequency * current_time + offset)
            target_positions.append(target_pos)

        return target_positions

    def calculate_end_effector_pose(self, joint_positions):
        """
        Calculate end-effector pose from joint positions using forward kinematics.
        This is a simplified implementation - in practice, use Isaac's kinematics.
        """
        # This would use actual forward kinematics in real implementation
        # For demonstration, return a simple calculation
        if len(joint_positions) >= 6:  # At least 6 DOF for position + orientation
            # Simplified FK calculation (not accurate, just for demonstration)
            x = 0.5 + 0.1 * np.sin(joint_positions[0])
            y = 0.0 + 0.1 * np.cos(joint_positions[1])
            z = 0.8 + 0.1 * np.sin(joint_positions[2])

            # Simple orientation
            quat = tf_transformations.quaternion_from_euler(
                joint_positions[3], joint_positions[4], joint_positions[5]
            )

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]

            return pose

        return None


def main(args=None):
    rclpy.init(args=args)

    # Create nodes
    perception_node = IsaacROSPerceptionNode()
    navigation_node = IsaacROSNavigateNode()
    manipulation_node = IsaacROSManipulationNode()

    # Create multi-threaded executor
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(perception_node)
    executor.add_node(navigation_node)
    executor.add_node(manipulation_node)

    try:
        # Set a goal for navigation (example)
        def set_example_goal():
            time.sleep(2)  # Wait for nodes to initialize
            goal_msg = PoseStamped()
            goal_msg.header.stamp = navigation_node.get_clock().now().to_msg()
            goal_msg.header.frame_id = 'map'
            goal_msg.pose.position.x = 2.0
            goal_msg.pose.position.y = 2.0
            goal_msg.pose.position.z = 0.0
            goal_msg.pose.orientation.w = 1.0

            navigation_node.goal = goal_msg
            navigation_node.get_logger().info('Example goal set: (2.0, 2.0)')

        # Start goal setting in a separate thread
        goal_thread = threading.Thread(target=set_example_goal)
        goal_thread.start()

        executor.spin()
    except KeyboardInterrupt:
        perception_node.get_logger().info('Shutting down Isaac ROS nodes...')
    finally:
        goal_thread.join(timeout=1.0)
        perception_node.destroy_node()
        navigation_node.destroy_node()
        manipulation_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
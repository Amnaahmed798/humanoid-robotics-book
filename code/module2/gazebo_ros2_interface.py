#!/usr/bin/env python3

"""
Gazebo-ROS2 Interface for Robot Control and Sensor Processing

This example demonstrates how to interface with Gazebo simulation through ROS 2,
including robot control, sensor data processing, and navigation with obstacle avoidance.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import numpy as np
import math
from tf2_ros import TransformBroadcaster
import tf_transformations


class GazeboROS2Interface(Node):
    """
    Interface node for controlling a robot in Gazebo simulation and processing sensor data.
    """

    def __init__(self):
        super().__init__('gazebo_ros2_interface')

        # QoS profile for sensor data
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/mobile_robot/cmd_vel',
            10
        )

        # Subscribers
        self.laser_subscription = self.create_subscription(
            LaserScan,
            '/mobile_robot/scan',
            self.laser_callback,
            qos_profile
        )

        self.odom_subscription = self.create_subscription(
            Odometry,
            '/mobile_robot/odom',
            self.odom_callback,
            10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Robot state variables
        self.current_pose = Pose()
        self.current_twist = Twist()
        self.laser_ranges = []
        self.laser_angle_min = 0.0
        self.laser_angle_max = 0.0
        self.laser_angle_increment = 0.0

        # Navigation parameters
        self.goal_x = 2.0  # Goal position (from the world file)
        self.goal_y = 2.0
        self.min_obstacle_distance = 0.5  # Minimum distance to obstacles
        self.linear_speed = 0.5
        self.angular_speed = 0.8
        self.arrival_threshold = 0.3  # Distance threshold for reaching goal

        # Obstacle avoidance state
        self.obstacle_detected = False
        self.avoidance_mode = False

        self.get_logger().info('Gazebo-ROS2 Interface initialized')

    def odom_callback(self, msg):
        """Process odometry data from Gazebo."""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

        # Log position for debugging
        self.get_logger().debug(
            f'Robot position: ({self.current_pose.position.x:.2f}, '
            f'{self.current_pose.position.y:.2f})'
        )

    def laser_callback(self, msg):
        """Process laser scan data from Gazebo."""
        self.laser_ranges = np.array(msg.ranges)
        self.laser_angle_min = msg.angle_min
        self.laser_angle_max = msg.angle_max
        self.laser_angle_increment = msg.angle_increment

        # Process laser data for obstacle detection
        self.detect_obstacles()

    def detect_obstacles(self):
        """Detect obstacles in front of the robot."""
        if len(self.laser_ranges) == 0:
            return

        # Get ranges in front of the robot (forward 90 degrees)
        center_idx = len(self.laser_ranges) // 2
        front_range_indices = slice(
            center_idx - len(self.laser_ranges) // 8,
            center_idx + len(self.laser_ranges) // 8
        )

        front_ranges = self.laser_ranges[front_range_indices]
        front_ranges = front_ranges[np.isfinite(front_ranges)]  # Remove invalid ranges

        if len(front_ranges) > 0:
            min_front_distance = np.min(front_ranges)
            self.obstacle_detected = min_front_distance < self.min_obstacle_distance

            if self.obstacle_detected:
                self.get_logger().info(
                    f'Obstacle detected! Distance: {min_front_distance:.2f}m'
                )
        else:
            self.obstacle_detected = False

    def calculate_goal_direction(self):
        """Calculate direction to the goal."""
        dx = self.goal_x - self.current_pose.position.x
        dy = self.goal_y - self.current_pose.position.y
        distance_to_goal = math.sqrt(dx**2 + dy**2)
        goal_angle = math.atan2(dy, dx)

        return distance_to_goal, goal_angle

    def control_loop(self):
        """Main control loop for navigation and obstacle avoidance."""
        if len(self.laser_ranges) == 0:
            return

        # Calculate direction to goal
        distance_to_goal, goal_angle = self.calculate_goal_direction()

        # Check if we've reached the goal
        if distance_to_goal < self.arrival_threshold:
            self.stop_robot()
            self.get_logger().info('Goal reached!')
            return

        cmd_vel = Twist()

        if self.obstacle_detected:
            # Obstacle avoidance behavior
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = self.angular_speed  # Turn to avoid obstacle
            self.avoidance_mode = True
            self.get_logger().info('Executing obstacle avoidance')
        else:
            # Navigate toward goal
            current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

            # Calculate angle difference to goal
            angle_diff = self.normalize_angle(goal_angle - current_yaw)

            # Set control commands
            if abs(angle_diff) > 0.2:  # Need to turn
                cmd_vel.angular.z = self.angular_speed * np.sign(angle_diff)
                cmd_vel.linear.x = 0.0
            else:  # Move forward toward goal
                cmd_vel.linear.x = self.linear_speed
                cmd_vel.angular.z = 0.0

            self.avoidance_mode = False

        # Publish command
        self.cmd_vel_publisher.publish(cmd_vel)

        # Log command for debugging
        self.get_logger().debug(
            f'Cmd: linear={cmd_vel.linear.x:.2f}, angular={cmd_vel.angular.z:.2f}'
        )

    def get_yaw_from_quaternion(self, quaternion):
        """Extract yaw angle from quaternion."""
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def stop_robot(self):
        """Stop the robot."""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel)

    def simple_navigation(self):
        """
        Alternative simple navigation method for demonstration.
        This method implements a basic wall-following behavior.
        """
        if len(self.laser_ranges) == 0:
            return

        cmd_vel = Twist()

        # Get ranges in different sectors
        n = len(self.laser_ranges)
        front_ranges = self.laser_ranges[n//2 - n//8 : n//2 + n//8]
        left_ranges = self.laser_ranges[:n//4]
        right_ranges = self.laser_ranges[3*n//4:]

        front_valid = front_ranges[np.isfinite(front_ranges)]
        left_valid = left_ranges[np.isfinite(left_ranges)]
        right_valid = right_ranges[np.isfinite(right_ranges)]

        min_front = np.min(front_valid) if len(front_valid) > 0 else float('inf')
        min_left = np.min(left_valid) if len(left_valid) > 0 else float('inf')
        min_right = np.min(right_valid) if len(right_valid) > 0 else float('inf')

        # Simple wall following behavior
        target_distance = 0.8

        if min_front < 0.5:  # Obstacle in front
            if min_left > min_right:
                # Turn left
                cmd_vel.angular.z = self.angular_speed
                cmd_vel.linear.x = 0.1
            else:
                # Turn right
                cmd_vel.angular.z = -self.angular_speed
                cmd_vel.linear.x = 0.1
        elif min_left < target_distance * 0.8:  # Too close to left wall
            # Turn right slightly
            cmd_vel.angular.z = -0.3
            cmd_vel.linear.x = self.linear_speed
        elif min_left > target_distance * 1.2:  # Too far from left wall
            # Turn left slightly
            cmd_vel.angular.z = 0.3
            cmd_vel.linear.x = self.linear_speed
        else:  # Follow the wall
            cmd_vel.angular.z = 0.0
            cmd_vel.linear.x = self.linear_speed

        self.cmd_vel_publisher.publish(cmd_vel)


class NavigationEvaluator(Node):
    """
    Node to evaluate navigation performance in simulation.
    """

    def __init__(self):
        super().__init__('navigation_evaluator')

        # Subscribe to robot odometry
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/mobile_robot/odom',
            self.odom_callback,
            10
        )

        # Timer for evaluation
        self.eval_timer = self.create_timer(1.0, self.evaluate_performance)

        # Track navigation metrics
        self.start_time = self.get_clock().now()
        self.start_position = None
        self.current_position = None
        self.path_length = 0.0
        self.previous_position = None

        # Goal position (from world file)
        self.goal_position = (2.0, 2.0)
        self.success_threshold = 0.3

        self.get_logger().info('Navigation evaluator initialized')

    def odom_callback(self, msg):
        """Update robot position from odometry."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.current_position = (x, y)

        if self.start_position is None:
            self.start_position = (x, y)
            self.previous_position = (x, y)

        # Update path length
        if self.previous_position is not None:
            dx = x - self.previous_position[0]
            dy = y - self.previous_position[1]
            dist = math.sqrt(dx**2 + dy**2)
            self.path_length += dist

        self.previous_position = (x, y)

    def evaluate_performance(self):
        """Evaluate navigation performance."""
        if self.current_position is None:
            return

        # Calculate distance to goal
        dx = self.current_position[0] - self.goal_position[0]
        dy = self.current_position[1] - self.goal_position[1]
        distance_to_goal = math.sqrt(dx**2 + dy**2)

        # Calculate time elapsed
        current_time = self.get_clock().now()
        time_elapsed = (current_time - self.start_time).nanoseconds / 1e9

        # Calculate straight-line distance from start to goal
        if self.start_position is not None:
            dx_start = self.goal_position[0] - self.start_position[0]
            dy_start = self.goal_position[1] - self.start_position[1]
            straight_line_distance = math.sqrt(dx_start**2 + dy_start**2)

            # Calculate efficiency
            efficiency = straight_line_distance / self.path_length if self.path_length > 0 else 0

            self.get_logger().info(
                f'Navigation status - '
                f'Time: {time_elapsed:.2f}s, '
                f'Distance to goal: {distance_to_goal:.2f}m, '
                f'Path length: {self.path_length:.2f}m, '
                f'Efficiency: {efficiency:.2f}'
            )

        # Check if goal is reached
        if distance_to_goal < self.success_threshold:
            self.get_logger().info('SUCCESS: Goal reached!')
            self.get_logger().info(
                f'Final metrics - '
                f'Time: {time_elapsed:.2f}s, '
                f'Path length: {self.path_length:.2f}m'
            )


def main(args=None):
    rclpy.init(args=args)

    # Create nodes
    interface_node = GazeboROS2Interface()
    evaluator_node = NavigationEvaluator()

    # Create executor and add nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(interface_node)
    executor.add_node(evaluator_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        interface_node.get_logger().info('Shutting down nodes...')
    finally:
        interface_node.destroy_node()
        evaluator_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
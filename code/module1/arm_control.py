#!/usr/bin/env python3

"""
Basic ROS 2 node for moving a robotic arm.

This example demonstrates:
- Creating a ROS 2 node
- Using JointTrajectory messages to control arm joints
- Basic trajectory execution
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
import time


class ArmController(Node):
    def __init__(self):
        super().__init__('arm_controller')

        # Create publisher for joint trajectory commands
        self.traj_publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',  # Standard trajectory controller topic
            10
        )

        # Arm joint names (example for a 3-DOF arm)
        self.joint_names = ['joint1', 'joint2', 'joint3']

        # Timer to send commands periodically
        self.timer = self.create_timer(5.0, self.send_trajectory_command)

        self.get_logger().info('Arm Controller node initialized')

    def send_trajectory_command(self):
        """Send a trajectory command to move the arm to a specific position."""
        traj_msg = JointTrajectory()

        # Set header
        traj_msg.header = Header()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.header.frame_id = 'base_link'

        # Set joint names
        traj_msg.joint_names = self.joint_names

        # Create trajectory point
        point = JointTrajectoryPoint()

        # Set target joint positions (in radians)
        # Example: Move to a specific configuration
        point.positions = [0.5, -0.3, 0.8]  # Joint angles in radians

        # Set velocities (optional, for smoother motion)
        point.velocities = [0.1, 0.1, 0.1]

        # Set accelerations (optional)
        point.accelerations = [0.05, 0.05, 0.05]

        # Set time from start (when this point should be reached)
        point.time_from_start.sec = 2  # Reach this point in 2 seconds
        point.time_from_start.nanosec = 0

        # Add the point to the trajectory
        traj_msg.points = [point]

        # Publish the trajectory
        self.traj_publisher.publish(traj_msg)
        self.get_logger().info(f'Published trajectory command: positions={point.positions}')


def main(args=None):
    rclpy.init(args=args)

    arm_controller = ArmController()

    try:
        rclpy.spin(arm_controller)
    except KeyboardInterrupt:
        arm_controller.get_logger().info('Shutting down Arm Controller node...')
    finally:
        arm_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3

"""
Service example for humanoid joint control in ROS 2.

This example demonstrates:
- Creating a service server for joint control
- Creating a service client to call the service
- Handling service requests and responses
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from example_interfaces.srv import SetBool
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
import time


class JointControlService(Node):
    """Service server that provides joint control functionality."""

    def __init__(self):
        super().__init__('joint_control_service')

        # Create service
        self.srv = self.create_service(
            SetBool,  # Using SetBool as a simple example; in practice, you'd create a custom service
            'control_joint',
            self.control_joint_callback,
            callback_group=ReentrantCallbackGroup()
        )

        # Publisher for joint trajectory commands
        self.traj_publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Store current joint states
        self.current_joint_states = JointState()

        # Subscriber for joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.get_logger().info('Joint Control Service initialized')

    def joint_state_callback(self, msg):
        """Update current joint states."""
        self.current_joint_states = msg

    def control_joint_callback(self, request, response):
        """Handle joint control requests."""
        self.get_logger().info(f'Received joint control request: {request.data}')

        # In a real implementation, you would parse the request and execute the control command
        # For this example, we'll just simulate the control action
        if request.data:  # If request is True, move to a specific position
            success = self.move_to_position([0.5, -0.3, 0.8])  # Example joint positions
        else:  # If request is False, move to home position
            success = self.move_to_home_position()

        response.success = success
        response.message = f'Joint control executed successfully: {success}'

        self.get_logger().info(f'Responding to joint control request: {response.success}')
        return response

    def move_to_position(self, positions):
        """Move joints to specified positions."""
        try:
            traj_msg = JointTrajectory()
            traj_msg.header = Header()
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            traj_msg.header.frame_id = 'base_link'

            # Define joint names for the example
            traj_msg.joint_names = [f'joint{i+1}' for i in range(len(positions))]

            # Create trajectory point
            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = [0.1] * len(positions)
            point.accelerations = [0.05] * len(positions)
            point.time_from_start.sec = 2
            point.time_from_start.nanosec = 0

            traj_msg.points = [point]

            # Publish the trajectory
            self.traj_publisher.publish(traj_msg)
            time.sleep(0.1)  # Brief pause to ensure message is sent

            return True
        except Exception as e:
            self.get_logger().error(f'Error moving to position: {e}')
            return False

    def move_to_home_position(self):
        """Move joints to home position."""
        home_positions = [0.0, 0.0, 0.0]  # Home position for 3 joints
        return self.move_to_position(home_positions)


class JointControlClient(Node):
    """Service client for calling the joint control service."""

    def __init__(self):
        super().__init__('joint_control_client')

        # Create client
        self.cli = self.create_client(SetBool, 'control_joint')

        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.get_logger().info('Joint Control Client initialized')

    def send_request(self, control_flag):
        """Send a request to the joint control service."""
        request = SetBool.Request()
        request.data = control_flag

        self.get_logger().info(f'Sending joint control request: {control_flag}')

        future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        try:
            response = future.result()
            self.get_logger().info(f'Service response: {response.success}, {response.message}')
            return response
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            return None


def main(args=None):
    rclpy.init(args=args)

    # Create service server and client nodes
    service_node = JointControlService()
    client_node = JointControlClient()

    # Create multi-threaded executor to handle both nodes
    executor = MultiThreadedExecutor()
    executor.add_node(service_node)
    executor.add_node(client_node)

    # Example: Send a few requests from the client
    def send_test_requests():
        """Send test requests to the service."""
        time.sleep(2)  # Wait a bit for everything to initialize

        # Send request to move to position
        response1 = client_node.send_request(True)
        if response1:
            print(f"Move to position result: {response1.success}")

        time.sleep(3)  # Wait for movement to complete

        # Send request to move to home position
        response2 = client_node.send_request(False)
        if response2:
            print(f"Move to home result: {response2.success}")

    # Run test requests in a separate thread
    import threading
    test_thread = threading.Thread(target=send_test_requests)
    test_thread.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        service_node.get_logger().info('Shutting down nodes...')
    finally:
        test_thread.join(timeout=1.0)  # Wait for test thread to finish
        service_node.destroy_node()
        client_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
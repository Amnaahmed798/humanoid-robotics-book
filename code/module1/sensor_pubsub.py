#!/usr/bin/env python3

"""
Publisher and Subscriber example for sensor data in ROS 2.

This example demonstrates:
- Creating a publisher that simulates sensor data
- Creating a subscriber that processes the sensor data
- Using custom message types for sensor data
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
import random


class SensorPublisher(Node):
    """Node that publishes simulated sensor data."""

    def __init__(self):
        super().__init__('sensor_publisher')

        # Create publisher for sensor data
        self.publisher = self.create_publisher(
            JointState,
            'sensor_data',
            10
        )

        # Timer to publish data periodically
        self.timer = self.create_timer(0.1, self.publish_sensor_data)  # 10 Hz

        # Initialize joint names
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']

        self.get_logger().info('Sensor Publisher node initialized')

    def publish_sensor_data(self):
        """Publish simulated sensor data."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Set joint names
        msg.name = self.joint_names

        # Generate simulated sensor data (random values for demonstration)
        msg.position = [random.uniform(-1.5, 1.5) for _ in self.joint_names]
        msg.velocity = [random.uniform(-0.5, 0.5) for _ in self.joint_names]
        msg.effort = [random.uniform(-10.0, 10.0) for _ in self.joint_names]

        # Publish the message
        self.publisher.publish(msg)

        # Log the published data
        self.get_logger().info(f'Published sensor data: positions={msg.position}')


class SensorSubscriber(Node):
    """Node that subscribes to and processes sensor data."""

    def __init__(self):
        super().__init__('sensor_subscriber')

        # Create subscriber for sensor data
        self.subscription = self.create_subscription(
            JointState,
            'sensor_data',
            self.sensor_callback,
            10
        )

        # Store the latest sensor data
        self.latest_data = None

        self.get_logger().info('Sensor Subscriber node initialized')

    def sensor_callback(self, msg):
        """Process incoming sensor data."""
        self.latest_data = msg

        # Process the sensor data (example: calculate some metrics)
        avg_position = sum(msg.position) / len(msg.position) if msg.position else 0.0
        max_velocity = max(msg.velocity) if msg.velocity else 0.0

        self.get_logger().info(
            f'Received sensor data: '
            f'avg_pos={avg_position:.3f}, '
            f'max_vel={max_velocity:.3f}'
        )

        # Additional processing could happen here
        self.process_sensor_data(msg)

    def process_sensor_data(self, msg):
        """Perform additional processing on sensor data."""
        # Example: Check for joint limits
        joint_limits = {
            'joint1': (-1.5, 1.5),
            'joint2': (-2.0, 2.0),
            'joint3': (-1.5, 1.5),
            'joint4': (-2.5, 2.5),
            'joint5': (-3.0, 3.0)
        }

        for i, joint_name in enumerate(msg.name):
            if i < len(msg.position):
                pos = msg.position[i]
                if joint_name in joint_limits:
                    min_pos, max_pos = joint_limits[joint_name]
                    if not (min_pos <= pos <= max_pos):
                        self.get_logger().warn(
                            f'Joint {joint_name} position {pos:.3f} exceeds limits '
                            f'[{min_pos:.3f}, {max_pos:.3f}]'
                        )


class SensorProcessor(Node):
    """Node that both publishes and subscribes to sensor data."""

    def __init__(self):
        super().__init__('sensor_processor')

        # Publisher for processed data
        self.processed_publisher = self.create_publisher(
            Float32MultiArray,
            'processed_sensor_data',
            10
        )

        # Subscriber for raw sensor data
        self.subscription = self.create_subscription(
            JointState,
            'sensor_data',
            self.process_and_forward,
            10
        )

        self.get_logger().info('Sensor Processor node initialized')

    def process_and_forward(self, msg):
        """Process incoming data and publish processed results."""
        # Example processing: calculate velocity derivatives (acceleration)
        if len(msg.velocity) > 0:
            # In a real system, you'd use proper differentiation
            # This is just a simplified example
            processed_data = Float32MultiArray()
            processed_data.data = [abs(v) for v in msg.velocity]  # Example: absolute velocities

            self.processed_publisher.publish(processed_data)
            self.get_logger().info(f'Published processed data: {processed_data.data}')


def main(args=None):
    rclpy.init(args=args)

    # Create nodes
    publisher_node = SensorPublisher()
    subscriber_node = SensorSubscriber()
    processor_node = SensorProcessor()

    # Create executor and add nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(publisher_node)
    executor.add_node(subscriber_node)
    executor.add_node(processor_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        publisher_node.get_logger().info('Shutting down nodes...')
    finally:
        publisher_node.destroy_node()
        subscriber_node.destroy_node()
        processor_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
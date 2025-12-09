#!/usr/bin/env python3

"""
Capstone Demo: Autonomous Humanoid Robot System

This module demonstrates the complete autonomous humanoid robot system
integrating all components developed in the previous modules:
- ROS 2 communication and architecture
- Isaac Sim for simulation and perception
- VSLAM and navigation
- Synthetic data generation
- Reinforcement learning
- Multi-modal interaction
- Isaac integration
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import Image, LaserScan, CameraInfo, Imu, JointState
from geometry_msgs.msg import Twist, Pose, Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np
import cv2
from cv_bridge import CvBridge
import json
import time
from typing import Dict, List, Optional, Tuple
import threading
import queue
import math


class CapstoneDemoNode(Node):
    """
    Main node for the capstone demonstration of the autonomous humanoid robot.
    Integrates all modules into a cohesive demonstration system.
    """

    def __init__(self):
        super().__init__('capstone_demo_node')

        # CV Bridge for image processing
        self.cv_bridge = CvBridge()

        # TF2 for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # State management
        self.robot_state = {
            'current_pose': Pose(),
            'current_twist': Twist(),
            'joint_states': JointState(),
            'battery_level': 100.0,
            'operational_mode': 'idle',
            'current_task': 'none',
            'task_progress': 0.0
        }

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.status_pub = self.create_publisher(String, '/capstone/status', 10)
        self.task_pub = self.create_publisher(String, '/capstone/task', 10)
        self.visualization_pub = self.create_publisher(MarkerArray, '/capstone/visualization', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        self.camera_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.camera_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/capstone/command',
            self.command_callback,
            10
        )

        # Internal state
        self.latest_image = None
        self.latest_lidar = None
        self.perception_results = []
        self.navigation_map = None
        self.demo_state = 'idle'  # idle, initializing, executing_task, completed
        self.current_demo_task = None

        # Demo parameters
        self.demo_tasks = [
            'basic_navigation',
            'object_detection',
            'grasp_task',
            'social_interaction',
            'complex_task_sequence'
        ]

        # Timers
        self.status_timer = self.create_timer(1.0, self.publish_status)
        self.demo_timer = self.create_timer(0.1, self.demo_control_loop)

        self.get_logger().info('Capstone Demo Node initialized')

    def odom_callback(self, msg):
        """
        Update robot pose from odometry.
        """
        self.robot_state['current_pose'] = msg.pose.pose
        self.robot_state['current_twist'] = msg.twist.twist

    def joint_state_callback(self, msg):
        """
        Update joint states.
        """
        self.robot_state['joint_states'] = msg

    def imu_callback(self, msg):
        """
        Update IMU data.
        """
        self.robot_state['imu_data'] = msg

    def lidar_callback(self, msg):
        """
        Update LiDAR data.
        """
        self.latest_lidar = msg

    def camera_callback(self, msg):
        """
        Update camera data.
        """
        try:
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def command_callback(self, msg):
        """
        Process high-level commands for the demo.
        """
        command = msg.data.lower().strip()

        self.get_logger().info(f'Received demo command: {command}')

        if command.startswith('start_demo'):
            self.start_demo(command)
        elif command == 'stop_demo':
            self.stop_demo()
        elif command == 'pause_demo':
            self.pause_demo()
        elif command == 'resume_demo':
            self.resume_demo()
        elif command in self.demo_tasks:
            self.execute_specific_task(command)
        else:
            self.get_logger().warn(f'Unknown demo command: {command}')

    def start_demo(self, command: str):
        """
        Start the capstone demonstration.
        """
        self.get_logger().info('Starting capstone demonstration...')

        # Parse command for specific demo
        if ':' in command:
            task_name = command.split(':')[1]
            if task_name in self.demo_tasks:
                self.execute_specific_task(task_name)
            else:
                self.get_logger().warn(f'Unknown demo task: {task_name}')
        else:
            # Run full demo sequence
            self.demo_state = 'executing_task'
            self.run_demo_sequence()

    def run_demo_sequence(self):
        """
        Run the complete demo sequence.
        """
        self.get_logger().info('Running full demo sequence...')

        # Demo sequence: navigation -> perception -> manipulation -> interaction
        demo_sequence = [
            ('navigation_demo', self.execute_navigation_demo),
            ('perception_demo', self.execute_perception_demo),
            ('manipulation_demo', self.execute_manipulation_demo),
            ('interaction_demo', self.execute_interaction_demo)
        ]

        # Execute each step in sequence
        for demo_name, demo_func in demo_sequence:
            if self.demo_state != 'executing_task':
                break  # Demo was stopped

            self.get_logger().info(f'Executing {demo_name}...')
            self.current_demo_task = demo_name

            # Publish task status
            task_msg = String()
            task_msg.data = f'EXECUTING:{demo_name}'
            self.task_pub.publish(task_msg)

            # Execute the demo function
            success = demo_func()

            if not success:
                self.get_logger().error(f'Demo {demo_name} failed')
                break

            # Small delay between demos
            time.sleep(2.0)

        self.get_logger().info('Demo sequence completed')
        self.demo_state = 'completed'

    def execute_navigation_demo(self) -> bool:
        """
        Execute navigation demonstration.
        """
        self.get_logger().info('Executing navigation demo...')

        # Define navigation goals
        goals = [
            (1.0, 0.0, 0.0),   # Move 1m forward
            (1.0, 1.0, 1.57),  # Move to (1,1) and rotate 90 degrees
            (0.0, 1.0, 3.14),  # Move back to (0,1) and rotate 180 degrees
            (0.0, 0.0, 0.0)    # Return to origin
        ]

        for i, (x, y, theta) in enumerate(goals):
            if self.demo_state != 'executing_task':
                return False

            self.get_logger().info(f'Navigating to goal {i+1}: ({x}, {y}, {theta})')

            # Create goal message
            goal_msg = PoseStamped()
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.header.frame_id = 'map'
            goal_msg.pose.position.x = x
            goal_msg.pose.position.y = y
            goal_msg.pose.position.z = 0.0

            # Convert theta to quaternion
            from math import sin, cos
            goal_msg.pose.orientation.z = sin(theta / 2.0)
            goal_msg.pose.orientation.w = cos(theta / 2.0)

            # Publish goal
            self.goal_pub.publish(goal_msg)

            # Wait for navigation to complete (simplified)
            # In real implementation, would wait for navigation feedback
            time.sleep(3.0)

        return True

    def execute_perception_demo(self) -> bool:
        """
        Execute perception demonstration.
        """
        self.get_logger().info('Executing perception demo...')

        # Process current camera image to detect objects
        if self.latest_image is not None:
            # Simulate object detection
            detected_objects = self.simulate_object_detection(self.latest_image)

            # Publish visualization of detected objects
            self.publish_object_visualization(detected_objects)

            self.get_logger().info(f'Detected {len(detected_objects)} objects')

            # Process LiDAR data to detect obstacles
            if self.latest_lidar is not None:
                obstacles = self.process_lidar_for_obstacles(self.latest_lidar)
                self.get_logger().info(f'Detected {len(obstacles)} obstacles')

                # Visualize obstacles
                self.publish_obstacle_visualization(obstacles)

        return True

    def execute_manipulation_demo(self) -> bool:
        """
        Execute manipulation demonstration.
        """
        self.get_logger().info('Executing manipulation demo...')

        # In a real robot, this would involve:
        # 1. Detecting graspable objects
        # 2. Planning manipulation trajectory
        # 3. Executing grasp/release actions

        # For simulation, we'll publish manipulation commands
        try:
            # Simulate detecting a graspable object
            graspable_object = self.find_graspable_object()

            if graspable_object:
                self.get_logger().info(f'Attempting to grasp object at {graspable_object}')

                # Move to object position
                approach_pose = Pose()
                approach_pose.position.x = graspable_object.x - 0.2  # 20cm before object
                approach_pose.position.y = graspable_object.y
                approach_pose.position.z = 0.0
                approach_pose.orientation.w = 1.0

                # Simulate manipulation commands
                self.simulate_manipulation_commands(approach_pose)

                # Grasp object
                self.simulate_grasp_command()

                # Move to drop-off location
                dropoff_pose = Pose()
                dropoff_pose.position.x = 0.5
                dropoff_pose.position.y = 0.5
                dropoff_pose.position.z = 0.0
                dropoff_pose.orientation.w = 1.0

                self.simulate_manipulation_commands(dropoff_pose)

                # Release object
                self.simulate_release_command()

                self.get_logger().info('Manipulation demo completed')

        except Exception as e:
            self.get_logger().error(f'Error in manipulation demo: {e}')
            return False

        return True

    def execute_interaction_demo(self) -> bool:
        """
        Execute social interaction demonstration.
        """
        self.get_logger().info('Executing interaction demo...')

        # In a real robot, this would involve:
        # 1. Person detection and tracking
        # 2. Natural language processing
        # 3. Speech synthesis
        # 4. Gesture generation

        # For simulation, we'll simulate interaction commands
        try:
            # Detect people in the environment
            people_positions = self.simulate_person_detection()

            if people_positions:
                # Approach closest person
                closest_person = min(people_positions, key=lambda p: math.sqrt(p.x**2 + p.y**2))

                self.get_logger().info(f'Approaching person at {closest_person}')

                # Navigate to person
                goal_msg = PoseStamped()
                goal_msg.header.stamp = self.get_clock().now().to_msg()
                goal_msg.header.frame_id = 'map'
                goal_msg.pose.position.x = closest_person.x - 0.5  # Stop 50cm away
                goal_msg.pose.position.y = closest_person.y
                goal_msg.pose.position.z = 0.0
                goal_msg.pose.orientation.w = 1.0

                self.goal_pub.publish(goal_msg)

                # Wait for approach
                time.sleep(3.0)

                # Simulate greeting
                self.simulate_greeting_interaction()

                self.get_logger().info('Interaction demo completed')

        except Exception as e:
            self.get_logger().error(f'Error in interaction demo: {e}')
            return False

        return True

    def simulate_object_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Simulate object detection in the image.
        """
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define colors to detect (red, blue, green)
        colors = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255])
        }

        detected_objects = []

        for color_name, (lower, upper) in colors.items():
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Convert pixel coordinates to approximate world coordinates
                    # This is a simplification - in reality, depth information would be needed
                    world_x = (center_x - image.shape[1] / 2) * 0.001  # Rough conversion
                    world_y = (center_y - image.shape[0] / 2) * 0.001  # Rough conversion

                    obj = {
                        'class': color_name,
                        'confidence': 0.8,
                        'bbox': [x, y, x + w, y + h],
                        'center': [center_x, center_y],
                        'world_position': Point(x=world_x, y=world_y, z=0.0)
                    }
                    detected_objects.append(obj)

        return detected_objects

    def process_lidar_for_obstacles(self, lidar_msg: LaserScan) -> List[Point]:
        """
        Process LiDAR data to detect obstacles.
        """
        obstacles = []

        # Define thresholds
        min_distance = 0.3  # Minimum distance to consider as obstacle
        max_distance = 3.0  # Maximum distance to consider

        for i, range_val in enumerate(lidar_msg.ranges):
            if min_distance <= range_val <= max_distance:
                # Convert polar to Cartesian coordinates
                angle = lidar_msg.angle_min + i * lidar_msg.angle_increment

                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)

                obstacle_point = Point(x=x, y=y, z=0.0)
                obstacles.append(obstacle_point)

        return obstacles

    def find_graspable_object(self) -> Optional[Point]:
        """
        Find a graspable object in the environment.
        """
        # In real implementation, this would use perception results
        # For simulation, return a placeholder
        if self.perception_results:
            # Return the first detected object as graspable
            for obj in self.perception_results:
                if obj['class'] in ['red', 'blue', 'green']:  # Color-coded objects
                    return obj['world_position']

        # If no objects detected, return a placeholder position
        return Point(x=0.8, y=0.0, z=0.0)

    def simulate_person_detection(self) -> List[Point]:
        """
        Simulate person detection.
        """
        # In real implementation, this would use person detection algorithms
        # For simulation, return some placeholder positions
        return [
            Point(x=1.0, y=0.5, z=0.0),
            Point(x=0.5, y=-0.5, z=0.0)
        ]

    def simulate_manipulation_commands(self, pose: Pose):
        """
        Simulate manipulation commands.
        """
        # This would send commands to the manipulation system in real implementation
        self.get_logger().info(f'Simulating manipulation command to pose: ({pose.position.x}, {pose.position.y}, {pose.position.z})')

    def simulate_grasp_command(self):
        """
        Simulate grasp command.
        """
        self.get_logger().info('Simulating grasp command')

    def simulate_release_command(self):
        """
        Simulate release command.
        """
        self.get_logger().info('Simulating release command')

    def simulate_greeting_interaction(self):
        """
        Simulate greeting interaction.
        """
        self.get_logger().info('Simulating greeting interaction')

        # In real implementation, this would:
        # 1. Generate speech ("Hello! How can I help you?")
        # 2. Generate gesture (wave)
        # 3. Wait for response
        # 4. Process response

        # For simulation, just log the action
        speech_msg = String()
        speech_msg.data = "Hello! I'm your autonomous humanoid assistant. How can I help you today?"

        # Publish to speech system (would be connected to TTS)
        # self.speech_pub.publish(speech_msg)

    def publish_object_visualization(self, objects: List[Dict]):
        """
        Publish visualization markers for detected objects.
        """
        marker_array = MarkerArray()

        for i, obj in enumerate(objects):
            # Create marker for object position
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'detected_objects'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position = obj['world_position']
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            # Color based on object class
            if obj['class'] == 'red':
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif obj['class'] == 'blue':
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            elif obj['class'] == 'green':
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0

            marker.color.a = 0.8

            marker_array.markers.append(marker)

            # Add label
            label_marker = Marker()
            label_marker.header = marker.header
            label_marker.ns = 'object_labels'
            label_marker.id = i + 1000
            label_marker.type = Marker.TEXT_VIEW_FACING
            label_marker.action = Marker.ADD

            label_marker.pose.position = obj['world_position']
            label_marker.pose.position.z += 0.2  # Raise label above object
            label_marker.pose.orientation.w = 1.0

            label_marker.text = f"{obj['class'].upper()}"
            label_marker.scale.z = 0.1
            label_marker.color.r = 1.0
            label_marker.color.g = 1.0
            label_marker.color.b = 1.0
            label_marker.color.a = 1.0

            marker_array.markers.append(label_marker)

        self.visualization_pub.publish(marker_array)

    def publish_obstacle_visualization(self, obstacles: List[Point]):
        """
        Publish visualization markers for detected obstacles.
        """
        marker_array = MarkerArray()

        for i, obstacle in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'obstacles'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position = obstacle
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.5  # Height of obstacle

            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 0.6

            marker_array.markers.append(marker)

        self.visualization_pub.publish(marker_array)

    def publish_status(self):
        """
        Publish system status.
        """
        status_msg = String()

        status_info = {
            'state': self.demo_state,
            'current_task': self.current_demo_task,
            'robot_pose': {
                'x': self.robot_state['current_pose'].position.x,
                'y': self.robot_state['current_pose'].position.y,
                'z': self.robot_state['current_pose'].position.z
            },
            'battery_level': self.robot_state['battery_level'],
            'operational_mode': self.robot_state['operational_mode'],
            'timestamp': time.time()
        }

        status_msg.data = json.dumps(status_info)
        self.status_pub.publish(status_msg)

    def demo_control_loop(self):
        """
        Main control loop for the demo.
        """
        # This runs at 10Hz to update demo state
        if self.demo_state == 'executing_task':
            # Check if current task is complete
            # In real implementation, this would check actual task completion
            pass

    def execute_specific_task(self, task_name: str):
        """
        Execute a specific demo task.
        """
        self.get_logger().info(f'Executing specific task: {task_name}')
        self.current_demo_task = task_name
        self.demo_state = 'executing_task'

        if task_name == 'basic_navigation':
            success = self.execute_navigation_demo()
        elif task_name == 'object_detection':
            success = self.execute_perception_demo()
        elif task_name == 'grasp_task':
            success = self.execute_manipulation_demo()
        elif task_name == 'social_interaction':
            success = self.execute_interaction_demo()
        elif task_name == 'complex_task_sequence':
            success = self.run_demo_sequence()
        else:
            self.get_logger().warn(f'Unknown task: {task_name}')
            success = False

        if success:
            self.get_logger().info(f'Task {task_name} completed successfully')
        else:
            self.get_logger().error(f'Task {task_name} failed')

        self.demo_state = 'completed'

    def stop_demo(self):
        """
        Stop the current demo.
        """
        self.get_logger().info('Stopping demo...')
        self.demo_state = 'idle'
        self.current_demo_task = None

        # Stop robot
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

    def pause_demo(self):
        """
        Pause the current demo.
        """
        self.get_logger().info('Pausing demo...')
        self.demo_state = 'paused'

    def resume_demo(self):
        """
        Resume the paused demo.
        """
        self.get_logger().info('Resuming demo...')
        if self.current_demo_task:
            self.demo_state = 'executing_task'
        else:
            self.demo_state = 'idle'


class DemoOrchestrator(Node):
    """
    Orchestrates the complete capstone demonstration.
    """

    def __init__(self):
        super().__init__('demo_orchestrator')

        # Publisher for demo commands
        self.command_pub = self.create_publisher(String, '/capstone/command', 10)

        # Timer for demo orchestration
        self.demo_timer = self.create_timer(5.0, self.orchestrate_demo)

        # Demo sequence state
        self.demo_sequence = [
            'start_demo:navigation_demo',
            'start_demo:perception_demo',
            'start_demo:manipulation_demo',
            'start_demo:interaction_demo',
            'start_demo:complex_task_sequence'
        ]
        self.current_demo_index = 0
        self.demo_running = False

        self.get_logger().info('Demo Orchestrator initialized')

    def orchestrate_demo(self):
        """
        Orchestrate the demo sequence.
        """
        if not self.demo_running and self.current_demo_index < len(self.demo_sequence):
            command = String()
            command.data = self.demo_sequence[self.current_demo_index]

            self.get_logger().info(f'Publishing demo command: {command.data}')
            self.command_pub.publish(command)

            self.demo_running = True
            self.current_demo_index += 1

    def start_full_demo(self):
        """
        Start the full demo sequence.
        """
        self.current_demo_index = 0
        self.demo_running = False
        self.get_logger().info('Full demo sequence started')


def main(args=None):
    """
    Main function to run the capstone demonstration.
    """
    rclpy.init(args=args)

    # Create nodes
    demo_node = CapstoneDemoNode()
    orchestrator_node = DemoOrchestrator()

    # Create multi-threaded executor
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(demo_node)
    executor.add_node(orchestrator_node)

    try:
        # Start full demo sequence
        orchestrator_node.start_full_demo()

        executor.spin()
    except KeyboardInterrupt:
        demo_node.get_logger().info('Shutting down capstone demo...')
        orchestrator_node.get_logger().info('Shutting down orchestrator...')
    finally:
        demo_node.destroy_node()
        orchestrator_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
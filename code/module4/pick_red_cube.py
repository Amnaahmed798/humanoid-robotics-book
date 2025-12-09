#!/usr/bin/env python3

"""
Command robot to pick up the red cube using natural language and execute in simulation.

This example demonstrates how to integrate voice-to-action capabilities with
robot control, using perception to identify objects and manipulation to interact with them.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.spatial import distance
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
import time


class RedCubePicker(Node):
    """
    Node that demonstrates picking up a red cube based on voice command.
    """

    def __init__(self):
        super().__init__('red_cube_picker')

        # CV Bridge for image processing
        self.cv_bridge = CvBridge()

        # TF2 for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers
        self.arm_command_pub = self.create_publisher(
            JointTrajectory,
            '/left_arm_controller/joint_trajectory',
            10
        )

        self.gripper_command_pub = self.create_publisher(
            Float64MultiArray,
            '/gripper_controller/commands',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/red_cube_picker/status',
            10
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/object_detector/detections',
            self.detection_callback,
            10
        )

        # Internal state
        self.latest_image = None
        self.camera_intrinsics = None
        self.camera_info = None
        self.detected_objects = []
        self.red_cube_pose = None
        self.red_cube_pixel = None

        # Robot parameters
        self.arm_joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                          'wrist_flex', 'wrist_roll', 'gripper_finger']
        self.gripper_closed_pos = [0.0]  # Fully closed
        self.gripper_open_pos = [0.8]    # Fully open

        # Processing parameters
        self.red_lower = np.array([0, 50, 50])    # Lower HSV for red
        self.red_upper = np.array([10, 255, 255]) # Upper HSV for red
        self.min_object_area = 500  # Minimum area for object detection
        self.approach_height = 0.15  # Height to approach object from above
        self.grasp_height = 0.05     # Height to grasp object
        self.retract_height = 0.20   # Height to retract after grasp

        # State machine
        self.state = 'waiting'  # waiting, detecting, approaching, grasping, done
        self.state_timer = self.create_timer(0.1, self.state_machine)

        self.get_logger().info('Red Cube Picker initialized')

    def image_callback(self, msg):
        """
        Process incoming camera images.
        """
        try:
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def camera_info_callback(self, msg):
        """
        Process camera info for intrinsic parameters.
        """
        self.camera_info = msg
        self.camera_intrinsics = np.array(msg.k).reshape(3, 3)

    def detection_callback(self, msg):
        """
        Process object detection results.
        """
        self.detected_objects = []
        for detection in msg.detections:
            # Check if this is a red cube based on classification
            if detection.results:
                for result in detection.results:
                    if result.hypothesis.class_id.lower() in ['cube', 'block', 'red']:
                        self.detected_objects.append(detection)
                        break

        # Look for red cube specifically
        self.find_red_cube()

    def find_red_cube(self):
        """
        Find red cube using both color detection and object detection.
        """
        if self.latest_image is None:
            return

        # Method 1: Use color detection to find red objects
        hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)

        # Create mask for red color
        mask1 = cv2.inRange(hsv, self.red_lower, self.red_upper)
        mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Find contours in the red mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_object_area and area > best_area:
                # Check if this contour is cube-like (has corners)
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

                # A cube would have 4 corners approximately
                if len(approx) >= 4 and len(approx) <= 6:  # Allow some variation
                    best_contour = contour
                    best_area = area

        if best_contour is not None:
            # Get the center of the red cube
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                self.red_cube_pixel = (cx, cy)

                # Convert pixel to 3D world coordinates using depth
                # This would require depth information in real implementation
                # For now, we'll use a placeholder
                self.red_cube_pose = self.pixel_to_world(cx, cy, 0.5)  # 0.5m depth as placeholder

                self.get_logger().info(f'Red cube detected at pixel: ({cx}, {cy})')

    def pixel_to_world(self, u, v, depth):
        """
        Convert pixel coordinates to world coordinates.
        """
        if self.camera_intrinsics is None:
            return None

        # Camera intrinsic parameters
        fx = self.camera_intrinsics[0, 0]
        fy = self.camera_intrinsics[1, 1]
        cx = self.camera_intrinsics[0, 2]
        cy = self.camera_intrinsics[1, 2]

        # Convert to world coordinates
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        # Create Pose object
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z

        # Simple orientation (gripper facing down)
        pose.orientation.w = 1.0  # No rotation

        return pose

    def state_machine(self):
        """
        Main state machine for the picking operation.
        """
        if self.state == 'waiting':
            if self.red_cube_pose is not None:
                self.get_logger().info('Red cube found! Moving to approach state.')
                self.state = 'approaching'
                self.publish_status('Approaching red cube')

        elif self.state == 'approaching':
            success = self.approach_cube()
            if success:
                self.state = 'grasping'
                self.publish_status('Grasping red cube')

        elif self.state == 'grasping':
            success = self.grasp_cube()
            if success:
                self.state = 'done'
                self.publish_status('Successfully picked up red cube!')

        elif self.state == 'done':
            # Stay in done state
            pass

    def approach_cube(self):
        """
        Approach the red cube.
        """
        if self.red_cube_pose is None:
            return False

        self.get_logger().info(f'Approaching cube at {self.red_cube_pose.position}')

        # Calculate approach position (above the cube)
        approach_pose = Pose()
        approach_pose.position.x = self.red_cube_pose.position.x
        approach_pose.position.y = self.red_cube_pose.position.y
        approach_pose.position.z = self.red_cube_pose.position.z + self.approach_height
        approach_pose.orientation.w = 1.0

        # Generate trajectory to approach position
        trajectory = self.generate_arm_trajectory_to_pose(approach_pose)

        # Execute trajectory
        self.arm_command_pub.publish(trajectory)

        # Wait for approach to complete (simulated)
        time.sleep(2.0)

        return True

    def grasp_cube(self):
        """
        Grasp the red cube.
        """
        if self.red_cube_pose is None:
            return False

        self.get_logger().info('Lowering arm to grasp cube')

        # Move down to cube level
        grasp_pose = Pose()
        grasp_pose.position.x = self.red_cube_pose.position.x
        grasp_pose.position.y = self.red_cube_pose.position.y
        grasp_pose.position.z = self.red_cube_pose.position.z + self.grasp_height
        grasp_pose.orientation.w = 1.0

        trajectory = self.generate_arm_trajectory_to_pose(grasp_pose)
        self.arm_command_pub.publish(trajectory)

        # Wait for arm to reach position
        time.sleep(1.0)

        # Close gripper
        self.close_gripper()

        # Wait for grip
        time.sleep(1.0)

        # Lift cube
        lift_pose = Pose()
        lift_pose.position.x = self.red_cube_pose.position.x
        lift_pose.position.y = self.red_cube_pose.position.y
        lift_pose.position.z = self.red_cube_pose.position.z + self.retract_height
        lift_pose.orientation.w = 1.0

        trajectory = self.generate_arm_trajectory_to_pose(lift_pose)
        self.arm_command_pub.publish(trajectory)

        return True

    def generate_arm_trajectory_to_pose(self, target_pose):
        """
        Generate a simple trajectory to reach the target pose.
        """
        trajectory = JointTrajectory()
        trajectory.joint_names = self.arm_joints

        # Create trajectory point
        point = JointTrajectoryPoint()

        # This is a simplified trajectory - in reality, inverse kinematics would be needed
        # For demonstration, we'll use placeholder joint positions
        point.positions = [0.0, -0.5, 0.5, 0.0, 0.0, 0.0]  # Placeholder values

        # Set timing
        point.time_from_start.sec = 2
        point.time_from_start.nanosec = 0

        trajectory.points = [point]
        return trajectory

    def close_gripper(self):
        """
        Close the gripper to grasp the cube.
        """
        command = Float64MultiArray()
        command.data = self.gripper_closed_pos
        self.gripper_command_pub.publish(command)
        self.get_logger().info('Gripper closed')

    def open_gripper(self):
        """
        Open the gripper to release the cube.
        """
        command = Float64MultiArray()
        command.data = self.gripper_open_pos
        self.gripper_command_pub.publish(command)
        self.get_logger().info('Gripper opened')

    def publish_status(self, status_text):
        """
        Publish status updates.
        """
        status_msg = String()
        status_msg.data = status_text
        self.status_pub.publish(status_msg)


class VoiceCommandHandler(Node):
    """
    Node that handles voice commands and triggers appropriate actions.
    """

    def __init__(self):
        super().__init__('voice_command_handler')

        # Subscriber for voice commands
        self.voice_command_sub = self.create_subscription(
            String,
            '/voice/command',
            self.voice_command_callback,
            10
        )

        # Publisher for triggering actions
        self.trigger_pub = self.create_publisher(
            String,
            '/action/trigger',
            10
        )

        self.get_logger().info('Voice Command Handler initialized')

    def voice_command_callback(self, msg):
        """
        Process voice commands.
        """
        command = msg.data.lower()

        self.get_logger().info(f'Received voice command: {command}')

        # Check if command is to pick up red cube
        if 'pick up the red cube' in command or 'grasp the red cube' in command:
            self.get_logger().info('Red cube pickup command detected')

            # Trigger the red cube picker
            trigger_msg = String()
            trigger_msg.data = 'pick_red_cube'
            self.trigger_pub.publish(trigger_msg)

        elif 'release the cube' in command or 'drop the cube' in command:
            self.get_logger().info('Cube release command detected')

            # Trigger cube release
            trigger_msg = String()
            trigger_msg.data = 'release_cube'
            self.trigger_pub.publish(trigger_msg)

        else:
            self.get_logger().info(f'Command not recognized: {command}')


def main(args=None):
    """
    Main function to run the red cube picking demonstration.
    """
    rclpy.init(args=args)

    # Create nodes
    cube_picker = RedCubePicker()
    voice_handler = VoiceCommandHandler()

    # Create executor and add nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(cube_picker)
    executor.add_node(voice_handler)

    try:
        executor.spin()
    except KeyboardInterrupt:
        cube_picker.get_logger().info('Shutting down red cube picker...')
    finally:
        cube_picker.destroy_node()
        voice_handler.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
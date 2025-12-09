#!/usr/bin/env python3

"""
Multi-Modal Input Integration for Robotics

This module demonstrates how to integrate multiple input modalities (speech, vision, gesture)
for enhanced human-robot interaction. It processes inputs from various sources and
combines them to understand user intent more accurately.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import Image, CameraInfo, Joy
from geometry_msgs.msg import Point, PoseStamped
from vision_msgs.msg import Detection2DArray, Detection2D
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
import speech_recognition as sr
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json
import mediapipe as mp
import tensorflow as tf


@dataclass
class MultiModalInput:
    """
    Data structure for multi-modal input events.
    """
    speech_text: Optional[str] = None
    speech_confidence: float = 0.0
    vision_objects: List[Dict] = None
    gesture_type: Optional[str] = None
    gesture_confidence: float = 0.0
    pointing_coordinate: Optional[Point] = None
    timestamp: float = 0.0


class MultiModalInputProcessor(Node):
    """
    Node that processes multi-modal inputs and fuses them for better understanding.
    """

    def __init__(self):
        super().__init__('multi_modal_input_processor')

        # CV Bridge for image processing
        self.cv_bridge = CvBridge()

        # Initialize MediaPipe for gesture recognition
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

        # Queues for different input modalities
        self.speech_queue = queue.Queue()
        self.vision_queue = queue.Queue()
        self.gesture_queue = queue.Queue()

        # Publishers
        self.fused_input_pub = self.create_publisher(
            String,
            '/multi_modal/fused_input',
            10
        )

        self.intent_pub = self.create_publisher(
            String,
            '/multi_modal/intent',
            10
        )

        self.visualization_pub = self.create_publisher(
            Image,
            '/multi_modal/visualization',
            10
        )

        # Subscribers
        self.speech_sub = self.create_subscription(
            String,
            '/speech_recognition/text',
            self.speech_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/object_detector/detections',
            self.detection_callback,
            10
        )

        self.joystick_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joystick_callback,
            10
        )

        # Internal state
        self.latest_image = None
        self.latest_detections = []
        self.latest_joystick = None
        self.speech_buffer = []
        self.vision_buffer = []
        self.gesture_buffer = []

        # Fusion parameters
        self.temporal_window = 2.0  # seconds to consider for fusion
        self.confidence_threshold = 0.6

        # Start processing threads
        self.fusion_thread = threading.Thread(target=self.fusion_loop, daemon=True)
        self.fusion_thread.start()

        self.get_logger().info('Multi-Modal Input Processor initialized')

    def speech_callback(self, msg):
        """
        Process speech input.
        """
        speech_event = {
            'text': msg.data,
            'confidence': 0.9,  # Would come from ASR confidence
            'timestamp': time.time()
        }
        self.speech_queue.put(speech_event)

    def image_callback(self, msg):
        """
        Process image input for gesture recognition.
        """
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image

            # Process for gesture recognition
            gesture_info = self.recognize_gestures(cv_image)

            if gesture_info:
                gesture_event = {
                    'gesture_type': gesture_info['type'],
                    'confidence': gesture_info['confidence'],
                    'coordinate': gesture_info.get('coordinate'),
                    'timestamp': time.time()
                }
                self.gesture_queue.put(gesture_event)

            # Publish visualization
            self.publish_visualization(cv_image)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def detection_callback(self, msg):
        """
        Process object detection results.
        """
        detection_event = {
            'objects': [self.detection_to_dict(detection) for detection in msg.detections],
            'timestamp': time.time()
        }
        self.vision_queue.put(detection_event)

    def joystick_callback(self, msg):
        """
        Process joystick input as proxy for gesture input.
        """
        # In real implementation, this would come from gesture recognition
        # For simulation, we'll use joystick to simulate pointing gestures
        if abs(msg.axes[0]) > 0.5 or abs(msg.axes[1]) > 0.5:
            pointing_event = {
                'gesture_type': 'pointing',
                'confidence': 0.8,
                'coordinate': Point(x=msg.axes[0], y=msg.axes[1], z=0.0),
                'timestamp': time.time()
            }
            self.gesture_queue.put(pointing_event)

    def detection_to_dict(self, detection):
        """
        Convert Detection2D message to dictionary.
        """
        result = {
            'class': detection.results[0].hypothesis.class_id if detection.results else 'unknown',
            'confidence': detection.results[0].hypothesis.score if detection.results else 0.0,
            'bbox': {
                'center_x': detection.bbox.center.x,
                'center_y': detection.bbox.center.y,
                'size_x': detection.bbox.size_x,
                'size_y': detection.bbox.size_y
            }
        }
        return result

    def recognize_gestures(self, image):
        """
        Recognize gestures using MediaPipe.
        """
        if image is None:
            return None

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on image
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                # Analyze gesture
                gesture_type, confidence, coordinate = self.analyze_gesture(hand_landmarks, image.shape)

                if gesture_type:
                    return {
                        'type': gesture_type,
                        'confidence': confidence,
                        'coordinate': coordinate
                    }

        return None

    def analyze_gesture(self, hand_landmarks, image_shape):
        """
        Analyze hand landmarks to determine gesture type.
        """
        # Extract landmark positions
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))

        if len(landmarks) < 21:
            return None, 0.0, None

        # Index finger tip (landmark 8)
        index_tip = landmarks[8]
        index_mcp = landmarks[5]  # Metacarpophalangeal joint

        # Thumb tip (landmark 4)
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]

        # Calculate distances
        index_extended = ((index_tip[1] - index_mcp[1]) < -0.1)  # Finger extended if tip higher than base
        thumb_index_distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)

        # Classify gesture
        if index_extended and thumb_index_distance < 0.05:
            # Pointing gesture
            screen_x = int(index_tip[0] * image_shape[1])
            screen_y = int(index_tip[1] * image_shape[0])
            coordinate = Point(x=float(screen_x), y=float(screen_y), z=0.0)
            return 'pointing', 0.9, coordinate
        elif index_extended and not (thumb_index_distance < 0.05):
            # Index finger extended, not pointing
            return 'index_finger', 0.7, None
        elif thumb_index_distance < 0.03:
            # OK gesture
            screen_x = int(thumb_tip[0] * image_shape[1])
            screen_y = int(thumb_tip[1] * image_shape[0])
            coordinate = Point(x=float(screen_x), y=float(screen_y), z=0.0)
            return 'ok_sign', 0.8, coordinate
        else:
            # Other gesture
            return 'other', 0.5, None

    def publish_visualization(self, image):
        """
        Publish visualization image with gesture overlays.
        """
        try:
            viz_msg = self.cv_bridge.cv2_to_imgmsg(image, encoding='bgr8')
            viz_msg.header.stamp = self.get_clock().now().to_msg()
            viz_msg.header.frame_id = 'camera_link'
            self.visualization_pub.publish(viz_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing visualization: {e}')

    def fusion_loop(self):
        """
        Main fusion loop that combines inputs from different modalities.
        """
        while rclpy.ok():
            # Process all queues
            self.process_speech_queue()
            self.process_vision_queue()
            self.process_gesture_queue()

            # Perform fusion if we have inputs
            fused_input = self.perform_fusion()

            if fused_input:
                # Publish fused input
                fused_msg = String()
                fused_msg.data = json.dumps(fused_input)
                self.fused_input_pub.publish(fused_msg)

                # Extract intent and publish
                intent = self.extract_intent(fused_input)
                if intent:
                    intent_msg = String()
                    intent_msg.data = intent
                    self.intent_pub.publish(intent_msg)

            time.sleep(0.05)  # 20Hz fusion rate

    def process_speech_queue(self):
        """
        Process speech events from queue.
        """
        while not self.speech_queue.empty():
            try:
                event = self.speech_queue.get_nowait()
                self.speech_buffer.append(event)

                # Keep only recent events
                current_time = time.time()
                self.speech_buffer = [
                    evt for evt in self.speech_buffer
                    if current_time - evt['timestamp'] <= self.temporal_window
                ]
            except queue.Empty:
                break

    def process_vision_queue(self):
        """
        Process vision events from queue.
        """
        while not self.vision_queue.empty():
            try:
                event = self.vision_queue.get_nowait()
                self.vision_buffer.append(event)

                # Keep only recent events
                current_time = time.time()
                self.vision_buffer = [
                    evt for evt in self.vision_buffer
                    if current_time - evt['timestamp'] <= self.temporal_window
                ]
            except queue.Empty:
                break

    def process_gesture_queue(self):
        """
        Process gesture events from queue.
        """
        while not self.gesture_queue.empty():
            try:
                event = self.gesture_queue.get_nowait()
                self.gesture_buffer.append(event)

                # Keep only recent events
                current_time = time.time()
                self.gesture_buffer = [
                    evt for evt in self.gesture_buffer
                    if current_time - evt['timestamp'] <= self.temporal_window
                ]
            except queue.Empty:
                break

    def perform_fusion(self) -> Optional[Dict]:
        """
        Perform multi-modal fusion to create a unified input representation.
        """
        current_time = time.time()

        # Get most recent events from each modality
        recent_speech = self.get_most_recent(self.speech_buffer, current_time)
        recent_vision = self.get_most_recent(self.vision_buffer, current_time)
        recent_gesture = self.get_most_recent(self.gesture_buffer, current_time)

        if not any([recent_speech, recent_vision, recent_gesture]):
            return None

        # Create fused input
        fused_input = MultiModalInput(
            speech_text=recent_speech['text'] if recent_speech else None,
            speech_confidence=recent_speech['confidence'] if recent_speech else 0.0,
            vision_objects=recent_vision['objects'] if recent_vision else [],
            gesture_type=recent_gesture['gesture_type'] if recent_gesture else None,
            gesture_confidence=recent_gesture['confidence'] if recent_gesture else 0.0,
            pointing_coordinate=recent_gesture['coordinate'] if recent_gesture else None,
            timestamp=current_time
        )

        return {
            'speech': {
                'text': fused_input.speech_text,
                'confidence': fused_input.speech_confidence
            },
            'vision': {
                'objects': fused_input.vision_objects
            },
            'gesture': {
                'type': fused_input.gesture_type,
                'confidence': fused_input.gesture_confidence,
                'pointing_coordinate': {
                    'x': fused_input.pointing_coordinate.x,
                    'y': fused_input.pointing_coordinate.y,
                    'z': fused_input.pointing_coordinate.z
                } if fused_input.pointing_coordinate else None
            },
            'timestamp': fused_input.timestamp
        }

    def get_most_recent(self, event_buffer, current_time):
        """
        Get the most recent event from buffer that's within temporal window.
        """
        if not event_buffer:
            return None

        # Find the most recent event
        most_recent = None
        for event in event_buffer:
            if current_time - event['timestamp'] <= self.temporal_window:
                if most_recent is None or event['timestamp'] > most_recent['timestamp']:
                    most_recent = event

        return most_recent

    def extract_intent(self, fused_input: Dict) -> Optional[str]:
        """
        Extract intent from fused multi-modal input.
        """
        speech_text = fused_input['speech']['text']
        gesture_type = fused_input['gesture']['type']
        vision_objects = fused_input['vision']['objects']

        if not speech_text:
            return None

        # Intent classification based on multi-modal input
        speech_lower = speech_text.lower()

        # Navigation intents
        if any(keyword in speech_lower for keyword in ['go to', 'move to', 'navigate to', 'go to the']):
            if gesture_type == 'pointing':
                return 'navigate_to_pointed_location'
            else:
                return 'navigate_to_said_location'

        # Manipulation intents
        elif any(keyword in speech_lower for keyword in ['pick up', 'grasp', 'take', 'lift', 'get']):
            if gesture_type == 'pointing' and vision_objects:
                # Combine pointing with object detection to identify target object
                return 'grasp_pointed_object'
            elif vision_objects:
                # Just use speech + vision to identify target object
                return 'grasp_mentioned_object'
            else:
                return 'grasp_said_object'

        # Detection/search intents
        elif any(keyword in speech_lower for keyword in ['find', 'look for', 'where is', 'locate']):
            if gesture_type == 'pointing':
                return 'find_object_at_pointed_location'
            else:
                return 'find_said_object'

        # Affirmation/interrupt intents
        elif any(keyword in speech_lower for keyword in ['yes', 'ok', 'okay', 'sure']):
            if gesture_type == 'ok_sign':
                return 'strong_affirmation'
            else:
                return 'affirmation'

        elif any(keyword in speech_lower for keyword in ['no', 'stop', 'cancel']):
            if gesture_type in ['stop', 'hand_raised']:
                return 'strong_negation'
            else:
                return 'negation'

        # Social interaction intents
        elif any(keyword in speech_lower for keyword in ['hello', 'hi', 'good morning', 'good evening']):
            return 'greeting'

        # Default
        else:
            return 'unknown_intent'


class MultiModalCommandInterpreter(Node):
    """
    Interprets multi-modal fused inputs as robot commands.
    """

    def __init__(self):
        super().__init__('multi_modal_command_interpreter')

        # Subscribers
        self.fused_input_sub = self.create_subscription(
            String,
            '/multi_modal/fused_input',
            self.fused_input_callback,
            10
        )

        self.intent_sub = self.create_subscription(
            String,
            '/multi_modal/intent',
            self.intent_callback,
            10
        )

        # Publishers for robot commands
        self.nav_goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        self.manipulation_cmd_pub = self.create_publisher(
            String,
            '/manipulation/command',
            10
        )

        self.speech_cmd_pub = self.create_publisher(
            String,
            '/tts/text',
            10
        )

        # Internal state
        self.current_context = {
            'last_objects': [],
            'last_locations': [],
            'last_gestures': []
        }

        self.get_logger().info('Multi-Modal Command Interpreter initialized')

    def fused_input_callback(self, msg):
        """
        Process fused multi-modal input.
        """
        try:
            fused_data = json.loads(msg.data)

            # Update context
            if fused_data['vision']['objects']:
                self.current_context['last_objects'] = fused_data['vision']['objects']
            if fused_data['gesture']['type']:
                self.current_context['last_gestures'].append(fused_data['gesture'])

            self.get_logger().info(f'Fused input received: {fused_data}')

        except json.JSONDecodeError:
            self.get_logger().error('Could not parse fused input data')

    def intent_callback(self, msg):
        """
        Process interpreted intent and execute appropriate robot behavior.
        """
        intent = msg.data
        self.get_logger().info(f'Interpreting intent: {intent}')

        if intent == 'navigate_to_pointed_location':
            self.execute_navigate_to_pointed_location()
        elif intent == 'navigate_to_said_location':
            self.execute_navigate_to_said_location()
        elif intent == 'grasp_pointed_object':
            self.execute_grasp_pointed_object()
        elif intent == 'grasp_mentioned_object':
            self.execute_grasp_mentioned_object()
        elif intent == 'grasp_said_object':
            self.execute_grasp_said_object()
        elif intent == 'find_object_at_pointed_location':
            self.execute_find_object_at_pointed_location()
        elif intent == 'find_said_object':
            self.execute_find_said_object()
        elif intent == 'greeting':
            self.execute_greeting()
        elif intent == 'affirmation':
            self.execute_affirmation()
        elif intent == 'negation':
            self.execute_negation()
        else:
            self.execute_unknown_intent()

    def execute_navigate_to_pointed_location(self):
        """
        Execute navigation to location pointed by user.
        """
        self.get_logger().info('Executing navigation to pointed location')

        # In real implementation, this would calculate the world coordinates
        # from the pointing gesture and send a navigation goal
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = 1.0  # Placeholder coordinates
        goal_pose.pose.position.y = 1.0
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0

        self.nav_goal_pub.publish(goal_pose.pose)

    def execute_navigate_to_said_location(self):
        """
        Execute navigation to location mentioned in speech.
        """
        self.get_logger().info('Executing navigation to said location')

        # This would parse the speech for location names and navigate there
        # For demonstration, use a default location
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = 2.0  # Placeholder coordinates
        goal_pose.pose.position.y = 0.0
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0

        self.nav_goal_pub.publish(goal_pose.pose)

    def execute_grasp_pointed_object(self):
        """
        Execute grasping of object pointed by user.
        """
        self.get_logger().info('Executing grasp of pointed object')

        # This would identify the object at the pointing location and grasp it
        cmd_msg = String()
        cmd_msg.data = 'grasp_object_at_pointed_location'
        self.manipulation_cmd_pub.publish(cmd_msg)

    def execute_grasp_mentioned_object(self):
        """
        Execute grasping of object mentioned in speech and detected in vision.
        """
        self.get_logger().info('Executing grasp of mentioned object')

        # This would match speech object with detected objects and grasp it
        cmd_msg = String()
        cmd_msg.data = 'grasp_mentioned_object'
        self.manipulation_cmd_pub.publish(cmd_msg)

    def execute_grasp_said_object(self):
        """
        Execute grasping of object mentioned in speech.
        """
        self.get_logger().info('Executing grasp of said object')

        cmd_msg = String()
        cmd_msg.data = 'grasp_said_object'
        self.manipulation_cmd_pub.publish(cmd_msg)

    def execute_find_object_at_pointed_location(self):
        """
        Execute search for object at pointed location.
        """
        self.get_logger().info('Executing search at pointed location')

        cmd_msg = String()
        cmd_msg.data = 'search_at_pointed_location'
        self.manipulation_cmd_pub.publish(cmd_msg)

    def execute_find_said_object(self):
        """
        Execute search for object mentioned in speech.
        """
        self.get_logger().info('Executing search for said object')

        cmd_msg = String()
        cmd_msg.data = 'search_for_object'
        self.manipulation_cmd_pub.publish(cmd_msg)

    def execute_greeting(self):
        """
        Execute greeting behavior.
        """
        self.get_logger().info('Executing greeting')

        greeting_msg = String()
        greeting_msg.data = 'Hello! How can I assist you today?'
        self.speech_cmd_pub.publish(greeting_msg)

    def execute_affirmation(self):
        """
        Execute affirmation acknowledgment.
        """
        self.get_logger().info('Executing affirmation acknowledgment')

        ack_msg = String()
        ack_msg.data = 'I understand and agree.'
        self.speech_cmd_pub.publish(ack_msg)

    def execute_negation(self):
        """
        Execute negation acknowledgment.
        """
        self.get_logger().info('Executing negation acknowledgment')

        ack_msg = String()
        ack_msg.data = 'I understand. How else can I help?'
        self.speech_cmd_pub.publish(ack_msg)

    def execute_unknown_intent(self):
        """
        Execute response to unknown intent.
        """
        self.get_logger().info('Executing unknown intent response')

        response_msg = String()
        response_msg.data = 'I\'m sorry, I didn\'t understand that. Could you please repeat?'
        self.speech_cmd_pub.publish(response_msg)


def main(args=None):
    """
    Main function to run the multi-modal input integration system.
    """
    rclpy.init(args=args)

    # Create nodes
    input_processor = MultiModalInputProcessor()
    command_interpreter = MultiModalCommandInterpreter()

    # Create multi-threaded executor
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(input_processor)
    executor.add_node(command_interpreter)

    try:
        executor.spin()
    except KeyboardInterrupt:
        input_processor.get_logger().info('Shutting down multi-modal input processor...')
        command_interpreter.get_logger().info('Shutting down command interpreter...')
    finally:
        input_processor.destroy_node()
        command_interpreter.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
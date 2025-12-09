# Natural Language Command Translation to ROS 2 Actions

## Introduction

Translating natural language commands to ROS 2 actions is a critical component of cognitive robotic systems. This chapter explores how to convert human language into executable robot behaviors using a combination of natural language processing, semantic parsing, and ROS 2 action interfaces. We'll cover both rule-based and machine learning approaches to achieve robust command translation.

## Understanding Natural Language Commands

### Command Structure Analysis

Natural language commands typically follow patterns that can be categorized:

```
[Action] [Object] [Location] [Constraints]
```

Examples:
- "Go to the kitchen" → Action: navigate, Location: kitchen
- "Pick up the red ball" → Action: grasp, Object: red ball
- "Bring me the cup from the table" → Action: fetch, Object: cup, Location: table

### Command Categories

1. **Navigation Commands**: Move to locations, follow directions
2. **Manipulation Commands**: Pick up, place, grasp objects
3. **Perception Commands**: Look for, detect, identify objects
4. **Communication Commands**: Speak, listen, display information
5. **Complex Commands**: Multi-step tasks requiring planning

## Rule-Based Command Parsing

### Pattern Matching Approach

```python
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class ActionType(Enum):
    NAVIGATE = "navigate"
    GRASP = "grasp"
    RELEASE = "release"
    DETECT = "detect"
    FOLLOW = "follow"
    SPEAK = "speak"
    LISTEN = "listen"


@dataclass
class ParsedCommand:
    action: ActionType
    object_name: Optional[str] = None
    location: Optional[str] = None
    quantity: Optional[int] = None
    attributes: List[str] = None
    original_text: str = ""


class RuleBasedParser:
    """
    Rule-based parser for translating natural language to robot commands.
    """

    def __init__(self):
        # Navigation patterns
        self.nav_patterns = [
            (r'go\s+to\s+(?:the\s+)?(\w+)', ActionType.NAVIGATE),
            (r'move\s+to\s+(?:the\s+)?(\w+)', ActionType.NAVIGATE),
            (r'navigate\s+to\s+(?:the\s+)?(\w+)', ActionType.NAVIGATE),
            (r'go\s+(forward|backward|left|right)', ActionType.NAVIGATE),
            (r'move\s+(forward|backward|left|right)', ActionType.NAVIGATE),
        ]

        # Manipulation patterns
        self.manipulation_patterns = [
            (r'(?:pick\s+up|grasp|grab|take|lift)\s+(?:the\s+)?(.+)', ActionType.GRASP),
            (r'(?:place|put|set|drop|release)\s+(?:down\s+)?(?:the\s+)?(.+)', ActionType.RELEASE),
            (r'pick\s+(?:up\s+)?(.+)', ActionType.GRASP),
        ]

        # Detection patterns
        self.detection_patterns = [
            (r'(?:find|locate|look\s+for|detect|search\s+for)\s+(?:the\s+)?(.+)', ActionType.DETECT),
            (r'where\s+is\s+(?:the\s+)?(.+)', ActionType.DETECT),
        ]

        # Follow patterns
        self.follow_patterns = [
            (r'follow\s+(me|him|her|the\s+\w+)', ActionType.FOLLOW),
            (r'come\s+with\s+(me|him|her)', ActionType.FOLLOW),
        ]

        # Speak patterns
        self.speak_patterns = [
            (r'say\s+(.+)', ActionType.SPEAK),
            (r'speak\s+(.+)', ActionType.SPEAK),
        ]

        # Object attributes
        self.attribute_patterns = [
            (r'(red|blue|green|yellow|purple|orange|pink|black|white|gray)', 'color'),
            (r'(small|large|big|tiny|huge|medium)', 'size'),
            (r'(square|round|rectangular|circular|triangular)', 'shape'),
        ]

    def parse_command(self, text: str) -> Optional[ParsedCommand]:
        """
        Parse natural language command into structured format.
        """
        text_lower = text.lower().strip()

        # Extract attributes first
        attributes = self.extract_attributes(text_lower)

        # Try navigation patterns
        for pattern, action_type in self.nav_patterns:
            match = re.search(pattern, text_lower)
            if match:
                location = match.group(1) if match.lastindex else None
                return ParsedCommand(
                    action=action_type,
                    location=location,
                    attributes=attributes,
                    original_text=text
                )

        # Try manipulation patterns
        for pattern, action_type in self.manipulation_patterns:
            match = re.search(pattern, text_lower)
            if match:
                object_name = match.group(1) if match.lastindex else None
                return ParsedCommand(
                    action=action_type,
                    object_name=object_name,
                    attributes=attributes,
                    original_text=text
                )

        # Try detection patterns
        for pattern, action_type in self.detection_patterns:
            match = re.search(pattern, text_lower)
            if match:
                object_name = match.group(1) if match.lastindex else None
                return ParsedCommand(
                    action=action_type,
                    object_name=object_name,
                    attributes=attributes,
                    original_text=text
                )

        # Try follow patterns
        for pattern, action_type in self.follow_patterns:
            match = re.search(pattern, text_lower)
            if match:
                target = match.group(1) if match.lastindex else None
                return ParsedCommand(
                    action=action_type,
                    object_name=target,
                    attributes=attributes,
                    original_text=text
                )

        # Try speak patterns
        for pattern, action_type in self.speak_patterns:
            match = re.search(pattern, text_lower)
            if match:
                message = match.group(1) if match.lastindex else None
                return ParsedCommand(
                    action=action_type,
                    object_name=message,  # Use object_name for message
                    attributes=attributes,
                    original_text=text
                )

        # If no pattern matched, return None
        return None

    def extract_attributes(self, text: str) -> List[str]:
        """
        Extract object attributes from text.
        """
        attributes = []

        for pattern, attr_type in self.attribute_patterns:
            matches = re.findall(pattern, text)
            attributes.extend(matches)

        return attributes

    def validate_parsed_command(self, parsed: ParsedCommand) -> bool:
        """
        Validate the parsed command for semantic correctness.
        """
        if not parsed.action:
            return False

        # Check if required fields are present based on action type
        if parsed.action in [ActionType.NAVIGATE, ActionType.FOLLOW]:
            if not parsed.location and not parsed.object_name:
                return False

        elif parsed.action in [ActionType.GRASP, ActionType.RELEASE, ActionType.DETECT]:
            if not parsed.object_name:
                return False

        elif parsed.action == ActionType.SPEAK:
            if not parsed.object_name:  # Using object_name for message
                return False

        return True


# Example usage
parser = RuleBasedParser()
command = parser.parse_command("Go to the kitchen and pick up the red ball")
if command:
    print(f"Action: {command.action}")
    print(f"Location: {command.location}")
    print(f"Object: {command.object_name}")
    print(f"Attributes: {command.attributes}")
```

## Semantic Parsing with Named Entity Recognition

### Advanced NER-based Parser

```python
import spacy
from typing import Tuple, Dict, Any
import json


class SemanticParser:
    """
    Semantic parser using spaCy for named entity recognition and dependency parsing.
    """

    def __init__(self):
        # Load spaCy model (install with: python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            # Fallback to basic regex if model not available
            self.nlp = None

    def parse_with_spacy(self, text: str) -> Optional[ParsedCommand]:
        """
        Parse command using spaCy NLP.
        """
        if not self.nlp:
            return None

        doc = self.nlp(text)

        # Extract action (verb)
        action = self.extract_action(doc)
        if not action:
            return None

        # Extract object
        obj = self.extract_object(doc)

        # Extract location
        location = self.extract_location(doc)

        # Extract attributes
        attributes = self.extract_attributes_from_doc(doc)

        return ParsedCommand(
            action=action,
            object_name=obj,
            location=location,
            attributes=attributes,
            original_text=text
        )

    def extract_action(self, doc) -> Optional[ActionType]:
        """
        Extract action from parsed document.
        """
        # Look for main verbs that indicate actions
        for token in doc:
            if token.pos_ == "VERB":
                lemma = token.lemma_.lower()

                # Map verb lemmas to action types
                if lemma in ["go", "move", "navigate", "walk", "run", "drive"]:
                    return ActionType.NAVIGATE
                elif lemma in ["pick", "grasp", "grab", "take", "lift", "hold"]:
                    return ActionType.GRASP
                elif lemma in ["place", "put", "set", "drop", "release", "lay"]:
                    return ActionType.RELEASE
                elif lemma in ["find", "locate", "look", "see", "detect", "search"]:
                    return ActionType.DETECT
                elif lemma in ["follow", "come", "track", "accompany"]:
                    return ActionType.FOLLOW
                elif lemma in ["say", "speak", "tell", "talk", "announce"]:
                    return ActionType.SPEAK

        return None

    def extract_object(self, doc) -> Optional[str]:
        """
        Extract object from parsed document.
        """
        for token in doc:
            # Look for direct objects
            if token.dep_ == "dobj":
                # Get the full noun phrase
                return self.get_full_noun_phrase(token)

            # Look for prepositional objects
            if token.dep_ == "pobj" and token.head.text in ["for", "up", "down", "at"]:
                return self.get_full_noun_phrase(token)

        # If no direct object, look for compound nouns
        for chunk in doc.noun_chunks:
            if chunk.root.dep_ == "dobj" or chunk.root.head.pos_ == "VERB":
                return chunk.text

        return None

    def extract_location(self, doc) -> Optional[str]:
        """
        Extract location from parsed document.
        """
        for token in doc:
            # Look for prepositional phrases indicating location
            if token.dep_ == "prep" and token.head.pos_ == "VERB":
                # Get the prepositional phrase
                prep_obj = [child for child in token.children if child.dep_ == "pobj"]
                if prep_obj:
                    return self.get_full_noun_phrase(prep_obj[0])

        # Look for noun chunks that might be locations
        for chunk in doc.noun_chunks:
            if chunk.text.lower() in ["kitchen", "living room", "bedroom", "office", "hallway", "door", "window"]:
                return chunk.text

        return None

    def extract_attributes_from_doc(self, doc) -> List[str]:
        """
        Extract attributes from parsed document.
        """
        attributes = []

        for token in doc:
            # Look for adjectives modifying nouns
            if token.pos_ == "ADJ":
                # Check if this adjective modifies a nearby noun
                for child in token.head.children:
                    if child.dep_ == "amod" and child.text == token.text:
                        attributes.append(token.text)

        return attributes

    def get_full_noun_phrase(self, token) -> str:
        """
        Get the full noun phrase starting from the token.
        """
        # Get all tokens in the subtree
        tokens = [t.text for t in token.subtree if t.pos_ in ["NOUN", "PROPN", "ADJ", "DET"]]
        return " ".join(tokens)
```

## Machine Learning Approach with Transformers

### Transformer-Based Command Classifier

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
import numpy as np


class TransformerCommandClassifier:
    """
    Transformer-based classifier for command categorization.
    """

    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Define number of action classes
        self.num_actions = len(ActionType)

        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_actions
        )

        # Action to index mapping
        self.action_to_idx = {action.value: idx for idx, action in enumerate(ActionType)}
        self.idx_to_action = {idx: action for action, idx in self.action_to_idx.items()}

    def preprocess_command(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess command text for transformer model.
        """
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        return encoded

    def classify_command(self, text: str) -> Tuple[ActionType, float]:
        """
        Classify command text into action type with confidence score.
        """
        inputs = self.preprocess_command(text)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_idx = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_idx].item()

        action = self.idx_to_action[predicted_idx]
        return action, confidence

    def extract_entities(self, text: str) -> Dict[str, str]:
        """
        Extract entities using a sequence labeling approach.
        """
        # This would typically use a NER model like BERT for NER
        # For simplicity, we'll use the rule-based approach here
        # In practice, you'd use a dedicated NER model

        parser = RuleBasedParser()
        parsed = parser.parse_command(text)

        if parsed:
            return {
                'object': parsed.object_name or '',
                'location': parsed.location or '',
                'attributes': ', '.join(parsed.attributes or [])
            }
        else:
            return {'object': '', 'location': '', 'attributes': ''}


class HybridCommandProcessor:
    """
    Hybrid processor combining rule-based and ML approaches.
    """

    def __init__(self):
        self.rule_parser = RuleBasedParser()
        self.ml_classifier = TransformerCommandClassifier()
        self.semantic_parser = SemanticParser() if self.is_spacy_available() else None

    def is_spacy_available(self) -> bool:
        """
        Check if spaCy is available.
        """
        try:
            import spacy
            return True
        except ImportError:
            return False

    def process_command(self, text: str) -> ParsedCommand:
        """
        Process command using hybrid approach.
        """
        # Try rule-based parsing first (fast and reliable for common patterns)
        rule_parsed = self.rule_parser.parse_command(text)
        if rule_parsed and self.rule_parser.validate_parsed_command(rule_parsed):
            return rule_parsed

        # Try semantic parsing if spaCy is available
        if self.semantic_parser:
            semantic_parsed = self.semantic_parser.parse_with_spacy(text)
            if semantic_parsed and self.rule_parser.validate_parsed_command(semantic_parsed):
                return semantic_parsed

        # Fall back to ML classification
        ml_action, confidence = self.ml_classifier.classify_command(text)

        if confidence > 0.7:  # Only use if confidence is high enough
            entities = self.ml_classifier.extract_entities(text)

            return ParsedCommand(
                action=ml_action,
                object_name=entities.get('object'),
                location=entities.get('location'),
                attributes=entities.get('attributes', '').split(', ') if entities.get('attributes') else [],
                original_text=text
            )

        # If all methods fail, return a default command
        return ParsedCommand(
            action=ActionType.SPEAK,
            object_name=f"I don't understand the command: {text}",
            original_text=text
        )
```

## ROS 2 Action Interface Integration

### Action Translation System

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from action_msgs.msg import GoalStatus

# Import common action interfaces
from nav2_msgs.action import NavigateToPose
from control_msgs.action import GripperCommand
from std_srvs.srv import Trigger


class NLCommandTranslator(Node):
    """
    Node that translates natural language commands to ROS 2 actions.
    """

    def __init__(self):
        super().__init__('nl_command_translator')

        # Initialize command processor
        self.command_processor = HybridCommandProcessor()

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/command_status', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String,
            '/natural_language_command',
            self.command_callback,
            10
        )

        # Action clients
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.gripper_client = ActionClient(self, GripperCommand, 'gripper_command')

        # Service clients
        self.speech_client = self.create_client(Trigger, 'speak_text')

        # Wait for action servers
        self.nav_client.wait_for_server()
        self.gripper_client.wait_for_server()

        self.get_logger().info('NL Command Translator initialized')

    def command_callback(self, msg):
        """
        Process natural language command.
        """
        command_text = msg.data
        self.get_logger().info(f'Received command: {command_text}')

        # Parse command
        parsed_command = self.command_processor.process_command(command_text)

        if parsed_command:
            self.get_logger().info(f'Parsed command: {parsed_command.action.value}')
            self.execute_parsed_command(parsed_command)
        else:
            self.get_logger().error(f'Could not parse command: {command_text}')
            self.send_error_response(f"Sorry, I couldn't understand: {command_text}")

    def execute_parsed_command(self, parsed_command: ParsedCommand):
        """
        Execute the parsed command based on its action type.
        """
        action_type = parsed_command.action

        if action_type == ActionType.NAVIGATE:
            self.execute_navigation_command(parsed_command)
        elif action_type == ActionType.GRASP:
            self.execute_grasp_command(parsed_command)
        elif action_type == ActionType.RELEASE:
            self.execute_release_command(parsed_command)
        elif action_type == ActionType.DETECT:
            self.execute_detection_command(parsed_command)
        elif action_type == ActionType.FOLLOW:
            self.execute_follow_command(parsed_command)
        elif action_type == ActionType.SPEAK:
            self.execute_speak_command(parsed_command)
        else:
            self.send_error_response(f"Unknown action type: {action_type}")

    def execute_navigation_command(self, parsed_command: ParsedCommand):
        """
        Execute navigation command.
        """
        location = parsed_command.location

        if not location:
            self.send_error_response("No destination specified for navigation")
            return

        # Map location names to coordinates (would come from map server in practice)
        location_coords = self.get_location_coordinates(location)

        if location_coords:
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = 'map'
            goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
            goal_msg.pose.pose.position.x = location_coords[0]
            goal_msg.pose.pose.position.y = location_coords[1]
            goal_msg.pose.pose.position.z = 0.0

            # Set orientation (facing forward)
            from math import pi
            goal_msg.pose.pose.orientation.z = 0.0
            goal_msg.pose.pose.orientation.w = 1.0

            self.get_logger().info(f'Navigating to {location} at ({location_coords[0]}, {location_coords[1]})')

            # Send navigation goal
            future = self.nav_client.send_goal_async(goal_msg)
            future.add_done_callback(self.navigation_done_callback)
        else:
            self.send_error_response(f"Unknown location: {location}")

    def get_location_coordinates(self, location_name: str) -> Optional[Tuple[float, float]]:
        """
        Get coordinates for a named location.
        """
        # This would normally come from a map server or location database
        location_map = {
            'kitchen': (-2.0, 1.0),
            'living room': (1.0, -1.0),
            'bedroom': (2.0, 2.0),
            'office': (-1.0, -2.0),
            'entrance': (0.0, 0.0)
        }

        return location_map.get(location_name.lower())

    def navigation_done_callback(self, future):
        """
        Callback when navigation goal is completed.
        """
        goal_handle = future.result()
        if goal_handle.accepted:
            self.get_logger().info('Navigation goal accepted')
        else:
            self.get_logger().error('Navigation goal rejected')

    def execute_grasp_command(self, parsed_command: ParsedCommand):
        """
        Execute grasp command.
        """
        object_name = parsed_command.object_name

        if not object_name:
            self.send_error_response("No object specified for grasping")
            return

        self.get_logger().info(f'Attempting to grasp: {object_name}')

        # First, navigate to object location
        # This would involve object detection and localization
        # For now, we'll simulate the process

        # Send gripper command
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = 0.0  # Closed position
        goal_msg.command.max_effort = 100.0

        self.get_logger().info('Sending grasp command')

        future = self.gripper_client.send_goal_async(goal_msg)
        future.add_done_callback(self.gripper_done_callback)

    def execute_release_command(self, parsed_command: ParsedCommand):
        """
        Execute release command.
        """
        object_name = parsed_command.object_name

        if not object_name:
            self.send_error_response("No object specified for releasing")
            return

        self.get_logger().info(f'Attempting to release: {object_name}')

        # Send gripper command to open
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = 1.0  # Open position
        goal_msg.command.max_effort = 100.0

        self.get_logger().info('Sending release command')

        future = self.gripper_client.send_goal_async(goal_msg)
        future.add_done_callback(self.gripper_done_callback)

    def gripper_done_callback(self, future):
        """
        Callback when gripper action is completed.
        """
        goal_handle = future.result()
        if goal_handle.accepted:
            self.get_logger().info('Gripper action accepted')
        else:
            self.get_logger().error('Gripper action rejected')

    def execute_detection_command(self, parsed_command: ParsedCommand):
        """
        Execute detection command.
        """
        object_name = parsed_command.object_name

        if not object_name:
            self.send_error_response("No object specified for detection")
            return

        self.get_logger().info(f'Looking for: {object_name}')

        # This would trigger object detection system
        # For now, we'll simulate the process
        detected = self.simulate_object_detection(object_name)

        if detected:
            response = f"I found the {object_name}!"
        else:
            response = f"I couldn't find the {object_name}."

        self.send_speech_response(response)

    def simulate_object_detection(self, object_name: str) -> bool:
        """
        Simulate object detection process.
        """
        # In real implementation, this would interface with perception system
        # For simulation, return True 70% of the time
        import random
        return random.random() < 0.7

    def execute_follow_command(self, parsed_command: ParsedCommand):
        """
        Execute follow command.
        """
        target = parsed_command.object_name or "person"

        self.get_logger().info(f'Following: {target}')

        # This would activate follow behavior
        # For now, send a simple command
        cmd = Twist()
        cmd.linear.x = 0.3  # Follow at 0.3 m/s
        cmd.angular.z = 0.0  # Don't turn

        # Publish follow command
        self.cmd_vel_pub.publish(cmd)

        self.send_speech_response(f"Now following {target}")

    def execute_speak_command(self, parsed_command: ParsedCommand):
        """
        Execute speak command.
        """
        message = parsed_command.object_name  # Using object_name for message

        if not message:
            self.send_error_response("No message to speak")
            return

        self.get_logger().info(f'Speaking: {message}')

        # Call speech service
        if self.speech_client.wait_for_service(timeout_sec=1.0):
            request = Trigger.Request()
            request.data = message
            future = self.speech_client.call_async(request)
            future.add_done_callback(self.speech_done_callback)
        else:
            self.get_logger().error('Speech service not available')

    def speech_done_callback(self, future):
        """
        Callback when speech is completed.
        """
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Speech completed successfully')
            else:
                self.get_logger().error(f'Speech failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Speech service call failed: {e}')

    def send_error_response(self, error_msg: str):
        """
        Send error response.
        """
        self.get_logger().error(error_msg)

        # Send error status
        status_msg = String()
        status_msg.data = f"ERROR: {error_msg}"
        self.status_pub.publish(status_msg)

        # Try to speak the error
        self.send_speech_response(f"Error: {error_msg}")

    def send_speech_response(self, message: str):
        """
        Send speech response.
        """
        if self.speech_client.wait_for_service(timeout_sec=1.0):
            request = Trigger.Request()
            request.data = message
            self.speech_client.call_async(request)
        else:
            self.get_logger().warn(f'Cannot speak: {message}')
```

## Context-Aware Command Processing

### Context Manager for Command Understanding

```python
class ContextManager:
    """
    Manages context for improved command understanding.
    """

    def __init__(self):
        self.conversation_history = []
        self.current_task = None
        self.robot_state = {}
        self.environment_state = {}
        self.user_preferences = {}

    def update_context(self, parsed_command: ParsedCommand, execution_result: str = ""):
        """
        Update context based on command and execution result.
        """
        context_entry = {
            'command': parsed_command,
            'timestamp': time.time(),
            'execution_result': execution_result
        }

        self.conversation_history.append(context_entry)

        # Keep only recent history (last 10 entries)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    def resolve_pronouns(self, text: str) -> str:
        """
        Resolve pronouns based on context.
        """
        # Simple pronoun resolution
        if 'it' in text.lower():
            # Find the last mentioned object
            for entry in reversed(self.conversation_history):
                if entry['command'].object_name:
                    resolved = text.lower().replace('it', entry['command'].object_name)
                    return resolved

        if 'there' in text.lower():
            # Find the last mentioned location
            for entry in reversed(self.conversation_history):
                if entry['command'].location:
                    resolved = text.lower().replace('there', entry['command'].location)
                    return resolved

        return text

    def infer_missing_information(self, parsed_command: ParsedCommand) -> ParsedCommand:
        """
        Infer missing information from context.
        """
        # If location is missing but context suggests one
        if not parsed_command.location and self.current_task:
            # For example, if current task is in kitchen, assume kitchen location
            if 'kitchen' in self.current_task:
                parsed_command.location = 'kitchen'

        # If object is generic, use context
        if parsed_command.object_name == 'it':
            for entry in reversed(self.conversation_history):
                if entry['command'].object_name:
                    parsed_command.object_name = entry['command'].object_name
                    break

        return parsed_command


class ContextAwareTranslator(NLCommandTranslator):
    """
    Command translator with context awareness.
    """

    def __init__(self):
        super().__init__()
        self.context_manager = ContextManager()

    def command_callback(self, msg):
        """
        Process natural language command with context awareness.
        """
        command_text = msg.data

        # Resolve pronouns and references
        resolved_text = self.context_manager.resolve_pronouns(command_text)
        self.get_logger().info(f'Resolved command: {resolved_text}')

        # Parse command
        parsed_command = self.command_processor.process_command(resolved_text)

        if parsed_command:
            # Infer missing information from context
            parsed_command = self.context_manager.infer_missing_information(parsed_command)

            self.get_logger().info(f'Parsed command: {parsed_command.action.value}')
            self.execute_parsed_command(parsed_command)

            # Update context
            self.context_manager.update_context(parsed_command, "executed")
        else:
            self.get_logger().error(f'Could not parse command: {resolved_text}')
            self.send_error_response(f"Sorry, I couldn't understand: {resolved_text}")
            self.context_manager.update_context(
                ParsedCommand(action=ActionType.SPEAK, object_name=f"Sorry, I couldn't understand: {resolved_text}"),
                "failed"
            )
```

## Validation and Error Handling

### Command Validation System

```python
class CommandValidator:
    """
    Validates commands for safety and feasibility.
    """

    def __init__(self):
        self.workspace_limits = {
            'x_min': -10.0, 'x_max': 10.0,
            'y_min': -10.0, 'y_max': 10.0,
            'z_min': 0.0, 'z_max': 2.0
        }

        self.payload_limits = {
            'max_weight': 1.0,  # kg
            'max_volume': 0.01   # m^3
        }

    def validate_command(self, parsed_command: ParsedCommand) -> Tuple[bool, str]:
        """
        Validate command for safety and feasibility.
        """
        action_type = parsed_command.action

        if action_type == ActionType.NAVIGATE:
            return self.validate_navigation_command(parsed_command)
        elif action_type == ActionType.GRASP:
            return self.validate_grasp_command(parsed_command)
        elif action_type == ActionType.RELEASE:
            return self.validate_release_command(parsed_command)
        else:
            # Other actions are generally safe
            return True, "Command is valid"

    def validate_navigation_command(self, parsed_command: ParsedCommand) -> Tuple[bool, str]:
        """
        Validate navigation command.
        """
        location = parsed_command.location

        if not location:
            return False, "No destination specified"

        # Check if location is within workspace limits
        coords = self.get_location_coordinates(location)
        if coords:
            x, y = coords
            if (x < self.workspace_limits['x_min'] or x > self.workspace_limits['x_max'] or
                y < self.workspace_limits['y_min'] or y > self.workspace_limits['y_max']):
                return False, f"Destination {location} is outside workspace limits"

        return True, "Navigation command is valid"

    def validate_grasp_command(self, parsed_command: ParsedCommand) -> Tuple[bool, str]:
        """
        Validate grasp command.
        """
        object_name = parsed_command.object_name

        if not object_name:
            return False, "No object specified for grasping"

        # In real implementation, check object properties (weight, size, etc.)
        # For now, assume all objects are graspable
        return True, "Grasp command is valid"

    def validate_release_command(self, parsed_command: ParsedCommand) -> Tuple[bool, str]:
        """
        Validate release command.
        """
        # Release commands are generally safe
        return True, "Release command is valid"

    def get_location_coordinates(self, location_name: str) -> Optional[Tuple[float, float]]:
        """
        Get coordinates for a named location (copied from NLCommandTranslator).
        """
        location_map = {
            'kitchen': (-2.0, 1.0),
            'living room': (1.0, -1.0),
            'bedroom': (2.0, 2.0),
            'office': (-1.0, -2.0),
            'entrance': (0.0, 0.0)
        }

        return location_map.get(location_name.lower())
```

## Performance Optimization

### Caching and Optimization

```python
from functools import lru_cache
import hashlib


class OptimizedCommandTranslator(ContextAwareTranslator):
    """
    Optimized command translator with caching and performance enhancements.
    """

    def __init__(self):
        super().__init__()
        self.command_cache = {}
        self.cache_size_limit = 100

    @lru_cache(maxsize=1000)
    def cached_parse_command(self, text: str) -> Optional[ParsedCommand]:
        """
        Cached command parsing for frequently used commands.
        """
        return self.command_processor.process_command(text)

    def command_callback(self, msg):
        """
        Process command with caching.
        """
        command_text = msg.data

        # Create cache key
        cache_key = hashlib.md5(command_text.encode()).hexdigest()

        # Check cache first
        if cache_key in self.command_cache:
            parsed_command = self.command_cache[cache_key]
            self.get_logger().info(f'Command found in cache: {command_text}')
        else:
            # Resolve pronouns and references
            resolved_text = self.context_manager.resolve_pronouns(command_text)

            # Parse command
            parsed_command = self.cached_parse_command(resolved_text)

            # Add to cache if successful
            if parsed_command:
                self.command_cache[cache_key] = parsed_command
                # Limit cache size
                if len(self.command_cache) > self.cache_size_limit:
                    # Remove oldest entry
                    oldest_key = next(iter(self.command_cache))
                    del self.command_cache[oldest_key]

        if parsed_command:
            # Validate command
            validator = CommandValidator()
            is_valid, validation_msg = validator.validate_command(parsed_command)

            if is_valid:
                # Infer missing information from context
                parsed_command = self.context_manager.infer_missing_information(parsed_command)

                self.get_logger().info(f'Parsed command: {parsed_command.action.value}')
                self.execute_parsed_command(parsed_command)

                # Update context
                self.context_manager.update_context(parsed_command, "executed")
            else:
                self.get_logger().error(f'Invalid command: {validation_msg}')
                self.send_error_response(f"Command not valid: {validation_msg}")
        else:
            self.get_logger().error(f'Could not parse command: {resolved_text}')
            self.send_error_response(f"Sorry, I couldn't understand: {resolved_text}")
            self.context_manager.update_context(
                ParsedCommand(action=ActionType.SPEAK, object_name=f"Sorry, I couldn't understand: {resolved_text}"),
                "failed"
            )


def main(args=None):
    rclpy.init(args=args)

    node = OptimizedCommandTranslator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down command translator...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Best Practices for NL to ROS 2 Translation

### Implementation Guidelines

1. **Layered Approach**: Use multiple parsing methods (rule-based, ML, semantic)
2. **Context Awareness**: Maintain conversation and task context
3. **Validation**: Always validate commands for safety and feasibility
4. **Error Handling**: Provide graceful fallbacks for unrecognized commands
5. **Caching**: Cache frequent commands to improve performance
6. **Feedback**: Provide clear feedback to users about command execution
7. **Security**: Sanitize inputs to prevent command injection
8. **Testing**: Extensively test with various command formulations

The natural language to ROS 2 action translation system enables robots to understand and respond to human commands in natural language, bridging the gap between human communication and robot execution capabilities.
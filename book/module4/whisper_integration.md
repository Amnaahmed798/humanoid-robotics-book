# Voice-to-Action with OpenAI Whisper Integration

## Introduction

Voice-to-action capabilities enable robots to understand and respond to spoken commands, making human-robot interaction more natural and intuitive. OpenAI's Whisper model provides state-of-the-art speech recognition that can be integrated into robotic systems to enable voice-controlled operation. This chapter covers the integration of Whisper with robotic systems for voice-to-action capabilities.

## Understanding Speech Recognition in Robotics

### Challenges in Robotic Speech Recognition

Speech recognition in robotics faces unique challenges compared to traditional applications:

1. **Acoustic Environment**: Robots operate in noisy environments with mechanical sounds
2. **Real-time Processing**: Commands need to be processed with minimal latency
3. **Domain Specificity**: Commands are often specific to robot capabilities
4. **Robustness**: Systems must handle various accents, speaking styles, and environmental conditions

### Whisper Model Overview

Whisper is a general-purpose speech recognition model that:
- Supports multiple languages
- Handles various accents and speaking styles
- Performs well in noisy environments
- Can be fine-tuned for specific domains
- Provides both speech-to-text and speech-to-speech capabilities

## Installing and Setting up Whisper

### Installation Requirements

```bash
# Install Whisper and related dependencies
pip install openai-whisper
pip install torch torchvision torchaudio  # PyTorch
pip install pyaudio  # For audio capture
pip install soundfile  # For audio file processing
pip install transformers  # For additional NLP processing
```

### Basic Whisper Usage

```python
import whisper
import torch

# Load the model
model = whisper.load_model("base")  # Options: tiny, base, small, medium, large

# Transcribe audio file
result = model.transcribe("path/to/audio.wav")
print(result["text"])
```

## Integrating Whisper with ROS 2

### Audio Capture and Processing Node

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
import pyaudio
import wave
import numpy as np
import whisper
import torch
import tempfile
import os
from threading import Thread, Lock
import queue


class WhisperAudioNode(Node):
    """
    ROS 2 node that captures audio and processes it with Whisper.
    """

    def __init__(self):
        super().__init__('whisper_audio_node')

        # Initialize Whisper model
        self.get_logger().info('Loading Whisper model...')
        self.model = whisper.load_model("base")
        self.get_logger().info('Whisper model loaded')

        # Audio parameters
        self.rate = 16000  # Sample rate
        self.chunk = 1024  # Buffer size
        self.format = pyaudio.paInt16
        self.channels = 1
        self.record_seconds = 3  # Duration of each recording

        # Publishers and subscribers
        self.transcript_pub = self.create_publisher(
            String,
            '/voice_transcript',
            10
        )

        self.command_pub = self.create_publisher(
            String,
            '/voice_command',
            10
        )

        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.processing_lock = Lock()

        # Start audio capture thread
        self.capture_thread = Thread(target=self.capture_audio, daemon=True)
        self.capture_thread.start()

        # Timer for processing audio
        self.process_timer = self.create_timer(0.1, self.process_audio_queue)

        self.get_logger().info('Whisper Audio Node initialized')

    def capture_audio(self):
        """
        Capture audio from microphone in a separate thread.
        """
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        self.get_logger().info('Audio capture started')

        while rclpy.ok():
            frames = []

            # Record for specified duration
            for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                data = stream.read(self.chunk)
                frames.append(data)

            # Add to processing queue
            self.audio_queue.put(frames)

        stream.stop_stream()
        stream.close()

    def process_audio_queue(self):
        """
        Process audio from queue using Whisper.
        """
        try:
            frames = self.audio_queue.get_nowait()
            self.process_audio_chunk(frames)
        except queue.Empty:
            pass  # No audio to process

    def process_audio_chunk(self, frames):
        """
        Process a chunk of audio frames with Whisper.
        """
        if self.processing_lock.locked():
            # Skip if already processing
            return

        with self.processing_lock:
            try:
                # Convert frames to numpy array
                audio_data = b''.join(frames)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)

                # Normalize audio
                audio_float = audio_np.astype(np.float32) / 32768.0

                # Save to temporary file for Whisper processing
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    # Write WAV file
                    wf = wave.open(temp_file.name, 'wb')
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.rate)
                    wf.writeframes(audio_data)
                    wf.close()

                    # Transcribe with Whisper
                    result = self.model.transcribe(temp_file.name)
                    transcript = result["text"].strip()

                    # Clean up temp file
                    os.unlink(temp_file.name)

                if transcript:
                    # Publish transcript
                    transcript_msg = String()
                    transcript_msg.data = transcript
                    self.transcript_pub.publish(transcript_msg)

                    self.get_logger().info(f'Heard: {transcript}')

                    # Process command if it's a robot command
                    self.process_command(transcript)

            except Exception as e:
                self.get_logger().error(f'Error processing audio: {e}')

    def process_command(self, transcript):
        """
        Process the transcript to extract robot commands.
        """
        # Convert to lowercase for easier processing
        text = transcript.lower()

        # Define command patterns
        command_patterns = {
            'move_forward': ['move forward', 'go forward', 'forward', 'move ahead', 'go ahead'],
            'move_backward': ['move backward', 'go backward', 'backward', 'back', 'reverse'],
            'turn_left': ['turn left', 'left', 'rotate left'],
            'turn_right': ['turn right', 'right', 'rotate right'],
            'stop': ['stop', 'halt', 'pause', 'freeze'],
            'pick_up': ['pick up', 'grasp', 'grab', 'take', 'lift'],
            'place_down': ['place down', 'put down', 'place', 'put', 'release'],
            'follow_me': ['follow me', 'follow', 'come with me'],
            'go_to': ['go to', 'move to', 'navigate to', 'go to location']
        }

        # Find matching command
        detected_command = None
        for command, patterns in command_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    detected_command = command
                    break
            if detected_command:
                break

        if detected_command:
            # Publish command
            cmd_msg = String()
            cmd_msg.data = detected_command
            self.command_pub.publish(cmd_msg)
            self.get_logger().info(f'Command detected: {detected_command}')
        else:
            self.get_logger().info('No command detected')

    def destroy_node(self):
        """
        Clean up audio resources.
        """
        self.audio.terminate()
        super().destroy_node()


class WhisperCommandInterpreter(Node):
    """
    Node to interpret Whisper commands and convert to robot actions.
    """

    def __init__(self):
        super().__init__('whisper_command_interpreter')

        # Subscribers
        self.transcript_sub = self.create_subscription(
            String,
            '/voice_transcript',
            self.transcript_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        # Publishers for robot control
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.gripper_cmd_pub = self.create_publisher(
            String,
            '/gripper_command',
            10
        )

        self.navigation_goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        # Command mapping
        self.command_mapping = {
            'move_forward': self.move_forward,
            'move_backward': self.move_backward,
            'turn_left': self.turn_left,
            'turn_right': self.turn_right,
            'stop': self.stop_robot,
            'pick_up': self.pick_up_object,
            'place_down': self.place_down_object,
            'follow_me': self.follow_mode,
            'go_to': self.go_to_location
        }

        self.get_logger().info('Whisper Command Interpreter initialized')

    def transcript_callback(self, msg):
        """
        Process transcript for context and intent analysis.
        """
        transcript = msg.data.lower()

        # Extract locations, objects, and other context from transcript
        self.extract_context(transcript)

    def command_callback(self, msg):
        """
        Execute the detected command.
        """
        command = msg.data

        if command in self.command_mapping:
            self.command_mapping[command]()
        else:
            self.get_logger().warn(f'Unknown command: {command}')

    def extract_context(self, transcript):
        """
        Extract context like locations, objects from transcript.
        """
        # This would use NLP techniques to extract entities
        # For example: "go to the kitchen" -> location: "kitchen"
        # "pick up the red ball" -> object: "red ball"
        pass

    def move_forward(self):
        """
        Move robot forward.
        """
        cmd = Twist()
        cmd.linear.x = 0.3  # m/s
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info('Moving forward')

    def move_backward(self):
        """
        Move robot backward.
        """
        cmd = Twist()
        cmd.linear.x = -0.3  # m/s
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info('Moving backward')

    def turn_left(self):
        """
        Turn robot left.
        """
        cmd = Twist()
        cmd.angular.z = 0.5  # rad/s
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info('Turning left')

    def turn_right(self):
        """
        Turn robot right.
        """
        cmd = Twist()
        cmd.angular.z = -0.5  # rad/s
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info('Turning right')

    def stop_robot(self):
        """
        Stop robot movement.
        """
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info('Stopping robot')

    def pick_up_object(self):
        """
        Command robot to pick up an object.
        """
        cmd = String()
        cmd.data = 'grasp'
        self.gripper_cmd_pub.publish(cmd)
        self.get_logger().info('Attempting to pick up object')

    def place_down_object(self):
        """
        Command robot to place down an object.
        """
        cmd = String()
        cmd.data = 'release'
        self.gripper_cmd_pub.publish(cmd)
        self.get_logger().info('Releasing object')

    def follow_mode(self):
        """
        Activate follow mode.
        """
        # This would activate a follow behavior
        self.get_logger().info('Activating follow mode')

    def go_to_location(self):
        """
        Navigate to a specific location.
        """
        # This would require location extraction from transcript
        self.get_logger().info('Navigating to location')
```

## Advanced Whisper Integration with Context

### Context-Aware Command Processing

```python
import re
from typing import Dict, List, Tuple


class ContextAwareWhisperProcessor:
    """
    Advanced Whisper processor that maintains context and state.
    """

    def __init__(self):
        self.context = {
            'current_location': 'unknown',
            'target_object': 'unknown',
            'task_sequence': [],
            'conversation_history': []
        }

    def process_transcript_with_context(self, transcript: str, current_context: Dict) -> Dict:
        """
        Process transcript with awareness of current context.
        """
        self.context.update(current_context)

        # Analyze the transcript
        analysis = self.analyze_transcript(transcript)

        # Determine intent based on context
        intent = self.determine_intent(transcript, analysis)

        # Extract entities considering context
        entities = self.extract_entities(transcript, intent)

        # Generate response/action
        action = self.generate_action(intent, entities)

        # Update conversation history
        self.context['conversation_history'].append({
            'transcript': transcript,
            'intent': intent,
            'entities': entities,
            'action': action
        })

        return {
            'intent': intent,
            'entities': entities,
            'action': action,
            'context': self.context.copy()
        }

    def analyze_transcript(self, transcript: str) -> Dict:
        """
        Analyze transcript for various linguistic features.
        """
        analysis = {
            'tokens': transcript.lower().split(),
            'entities': self.extract_named_entities(transcript),
            'action_verbs': self.extract_action_verbs(transcript),
            'spatial_prepositions': self.extract_spatial_prepositions(transcript),
            'quantifiers': self.extract_quantifiers(transcript)
        }

        return analysis

    def extract_named_entities(self, transcript: str) -> List[str]:
        """
        Extract named entities from transcript.
        """
        # Simple pattern matching for common robot entities
        entities = []

        # Object patterns
        object_patterns = [
            r'\b(red|blue|green|yellow|big|small|large|tiny)\s+(\w+)\b',
            r'\b(ball|cube|box|cup|bottle|chair|table)\b'
        ]

        for pattern in object_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    entities.append(' '.join(match))
                else:
                    entities.append(match)

        return entities

    def extract_action_verbs(self, transcript: str) -> List[str]:
        """
        Extract action verbs from transcript.
        """
        action_verbs = [
            'go', 'move', 'navigate', 'walk', 'run', 'turn', 'rotate',
            'pick', 'grasp', 'grab', 'take', 'lift', 'place', 'put', 'release',
            'find', 'locate', 'look', 'see', 'follow', 'come', 'bring'
        ]

        found_verbs = []
        tokens = transcript.lower().split()

        for token in tokens:
            for verb in action_verbs:
                if verb in token:
                    found_verbs.append(verb)

        return found_verbs

    def extract_spatial_prepositions(self, transcript: str) -> List[str]:
        """
        Extract spatial prepositions from transcript.
        """
        prepositions = [
            'to', 'at', 'in', 'on', 'under', 'over', 'next to', 'near', 'by',
            'left', 'right', 'front', 'back', 'behind', 'ahead'
        ]

        found_prep = []
        lower_transcript = transcript.lower()

        for prep in prepositions:
            if prep in lower_transcript:
                found_prep.append(prep)

        return found_prep

    def determine_intent(self, transcript: str, analysis: Dict) -> str:
        """
        Determine the intent of the command based on analysis.
        """
        transcript_lower = transcript.lower()

        # Intent classification based on keywords and context
        if any(verb in transcript_lower for verb in ['go', 'move', 'navigate', 'walk']):
            if any(prep in transcript_lower for prep in ['to', 'at', 'in']):
                return 'navigation_to_location'
            else:
                return 'simple_navigation'

        elif any(verb in transcript_lower for verb in ['pick', 'grasp', 'grab', 'take', 'lift']):
            return 'object_manipulation_pick'

        elif any(verb in transcript_lower for verb in ['place', 'put', 'release']):
            return 'object_manipulation_place'

        elif any(verb in transcript_lower for verb in ['find', 'locate', 'look', 'see']):
            return 'object_search'

        elif any(verb in transcript_lower for verb in ['follow', 'come']):
            return 'following'

        else:
            return 'unknown'

    def extract_entities(self, transcript: str, intent: str) -> Dict:
        """
        Extract entities relevant to the determined intent.
        """
        entities = {}

        if intent in ['navigation_to_location', 'object_search']:
            # Extract location names
            location_patterns = [
                r'to the (\w+)',
                r'at the (\w+)',
                r'in the (\w+)',
                r'to (\w+)',
                r'go (?:to|at|in) the? (\w+)'
            ]

            for pattern in location_patterns:
                match = re.search(pattern, transcript, re.IGNORECASE)
                if match:
                    entities['location'] = match.group(1)
                    break

        if intent in ['object_manipulation_pick', 'object_manipulation_place', 'object_search']:
            # Extract object descriptions
            object_patterns = [
                r'(?:the|a|an) ((?:red|blue|green|yellow|big|small|large|tiny)\s+\w+)',
                r'(?:the|a|an) (\w+)',
                r'(\w+) (?:on|at|near) the'
            ]

            for pattern in object_patterns:
                match = re.search(pattern, transcript, re.IGNORECASE)
                if match:
                    entities['object'] = match.group(1)
                    break

        return entities

    def generate_action(self, intent: str, entities: Dict) -> Dict:
        """
        Generate robot action based on intent and entities.
        """
        action = {
            'type': intent,
            'parameters': entities,
            'confidence': 0.9  # Default high confidence
        }

        # Add specific parameters based on intent
        if intent == 'navigation_to_location':
            action['ros_action'] = 'nav2_msgs.action.NavigateToPose'
            action['topic'] = '/navigate_to_pose'

        elif intent == 'object_manipulation_pick':
            action['ros_action'] = 'control_msgs.action.GripperCommand'
            action['topic'] = '/gripper_command'

        elif intent == 'object_manipulation_place':
            action['ros_action'] = 'control_msgs.action.GripperCommand'
            action['topic'] = '/gripper_command'

        elif intent == 'object_search':
            action['ros_action'] = 'object_search_action'
            action['topic'] = '/object_search'

        elif intent == 'following':
            action['ros_action'] = 'follow_action'
            action['topic'] = '/follow_person'

        return action
```

## Optimizing Whisper for Real-time Robotics

### Performance Optimization Techniques

```python
import asyncio
import concurrent.futures
from functools import partial
import time


class OptimizedWhisperProcessor:
    """
    Optimized Whisper processor for real-time robotics applications.
    """

    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.model = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.load_model()

    def load_model(self):
        """
        Load Whisper model with optimizations.
        """
        import whisper
        import torch

        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.get_logger().info(f'Loading Whisper model on {device}...')
        self.model = whisper.load_model(self.model_size).to(device)

        # Set to evaluation mode
        self.model.eval()

    def transcribe_audio_optimized(self, audio_path: str) -> str:
        """
        Optimized transcription with reduced latency.
        """
        # Use faster processing options
        result = self.model.transcribe(
            audio_path,
            fp16=torch.cuda.is_available(),  # Use float16 on GPU
            language='en',
            task='transcribe'
        )

        return result["text"]

    async def process_audio_async(self, audio_data):
        """
        Process audio asynchronously to avoid blocking.
        """
        loop = asyncio.get_event_loop()

        # Run transcription in thread pool to avoid blocking
        result = await loop.run_in_executor(
            self.executor,
            partial(self.transcribe_audio_optimized, audio_data)
        )

        return result

    def process_with_timeout(self, audio_path: str, timeout: float = 5.0) -> str:
        """
        Process audio with timeout to ensure real-time performance.
        """
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Whisper processing timed out")

        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))

        try:
            result = self.transcribe_audio_optimized(audio_path)
            signal.alarm(0)  # Cancel alarm
            return result
        except TimeoutError:
            self.get_logger().warn('Whisper processing timed out')
            return ""
        finally:
            signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
```

## Voice Command Grammar and Validation

### Command Validation System

```python
import json
import re
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class VoiceCommand:
    """
    Data structure for validated voice commands.
    """
    intent: str
    entities: Dict[str, str]
    confidence: float
    raw_transcript: str
    timestamp: float


class VoiceCommandValidator:
    """
    Validates voice commands against a defined grammar.
    """

    def __init__(self):
        self.command_grammar = self.load_command_grammar()

    def load_command_grammar(self) -> Dict:
        """
        Load the command grammar definition.
        """
        return {
            "navigation": {
                "patterns": [
                    r"^(?:go|move|navigate|walk)\s+(?:to|toward|towards)\s+(?:the\s+)?(\w+)$",
                    r"^(?:go|move|navigate|walk)\s+(forward|backward|left|right|ahead)$",
                    r"^(?:turn|rotate)\s+(left|right)$"
                ],
                "required_entities": ["destination"],
                "optional_entities": ["direction", "speed"]
            },
            "manipulation": {
                "patterns": [
                    r"^(?:pick|grasp|grab|take|lift)\s+(?:up\s+)?(?:the\s+)?(.+)$",
                    r"^(?:place|put|set|release)\s+(?:down\s+)?(?:the\s+)?(.+)(?:\s+on\s+(.+))?$"
                ],
                "required_entities": ["object"],
                "optional_entities": ["location"]
            },
            "query": {
                "patterns": [
                    r"^(?:where|what)\s+(?:is|are)\s+(?:the\s+)?(.+)$",
                    r"^(?:can\s+you|could\s+you)\s+(.+)\\?$"
                ],
                "required_entities": ["query_subject"],
                "optional_entities": []
            }
        }

    def validate_command(self, transcript: str) -> Optional[VoiceCommand]:
        """
        Validate transcript against command grammar.
        """
        transcript_lower = transcript.lower().strip()

        for intent, grammar in self.command_grammar.items():
            for pattern in grammar["patterns"]:
                match = re.match(pattern, transcript_lower)
                if match:
                    # Extract entities
                    entities = {}
                    if match.groups():
                        # This is a simplified entity extraction
                        # In practice, you'd map group indices to entity types
                        entities = {"extracted": match.group(1)}

                    return VoiceCommand(
                        intent=intent,
                        entities=entities,
                        confidence=0.85,  # Assume high confidence for grammatically correct commands
                        raw_transcript=transcript,
                        timestamp=time.time()
                    )

        # If no pattern matches, return None (invalid command)
        return None

    def suggest_corrections(self, invalid_transcript: str) -> List[str]:
        """
        Suggest possible corrections for invalid commands.
        """
        suggestions = []

        # Simple suggestions based on common mistakes
        if "g" in invalid_transcript.lower():
            # Suggest "go" if "g" is detected
            corrected = invalid_transcript.lower().replace("g ", "go ")
            if corrected != invalid_transcript.lower():
                suggestions.append(f"Did you mean: {corrected}?")

        # Add more sophisticated suggestion logic here

        return suggestions
```

## Integration with Robot Systems

### Complete Integration Example

```python
#!/usr/bin/env python3

"""
Complete integration of Whisper with robot control system.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import AudioData
import pyaudio
import wave
import numpy as np
import whisper
import torch
import tempfile
import os
import queue
from threading import Thread, Lock
import time


class IntegratedWhisperRobot(Node):
    """
    Complete integration of Whisper with robot control.
    """

    def __init__(self):
        super().__init__('integrated_whisper_robot')

        # Initialize Whisper model
        self.get_logger().info('Loading Whisper model...')
        self.model = whisper.load_model("base")
        self.get_logger().info('Whisper model loaded')

        # Audio parameters
        self.rate = 16000
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.record_seconds = 2

        # Publishers
        self.voice_cmd_pub = self.create_publisher(String, '/voice_command', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.nav_goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.processing_lock = Lock()

        # Robot state
        self.robot_state = {
            'current_location': 'unknown',
            'battery_level': 100,
            'gripper_status': 'open'  # or 'closed'
        }

        # Start audio capture
        self.capture_thread = Thread(target=self.capture_audio, daemon=True)
        self.capture_thread.start()

        # Timer for processing
        self.process_timer = self.create_timer(0.1, self.process_audio)

        self.get_logger().info('Integrated Whisper Robot initialized')

    def capture_audio(self):
        """
        Capture audio from microphone.
        """
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        self.get_logger().info('Audio capture started')

        while rclpy.ok():
            frames = []
            for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                data = stream.read(self.chunk)
                frames.append(data)

            self.audio_queue.put(frames)

        stream.stop_stream()
        stream.close()

    def process_audio(self):
        """
        Process audio from queue.
        """
        try:
            frames = self.audio_queue.get_nowait()
            self.process_audio_chunk(frames)
        except queue.Empty:
            pass

    def process_audio_chunk(self, frames):
        """
        Process audio chunk with Whisper.
        """
        if self.processing_lock.locked():
            return

        with self.processing_lock:
            try:
                # Convert to audio file for Whisper
                audio_data = b''.join(frames)

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    wf = wave.open(temp_file.name, 'wb')
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)
                    wf.setframerate(self.rate)
                    wf.writeframes(audio_data)
                    wf.close()

                    # Transcribe
                    result = self.model.transcribe(temp_file.name)
                    transcript = result["text"].strip()

                    # Clean up
                    os.unlink(temp_file.name)

                if transcript:
                    self.get_logger().info(f'Heard: {transcript}')

                    # Process command
                    self.execute_voice_command(transcript)

            except Exception as e:
                self.get_logger().error(f'Error processing audio: {e}')

    def execute_voice_command(self, transcript):
        """
        Execute voice command based on transcript.
        """
        # Convert to lowercase for processing
        cmd_lower = transcript.lower()

        # Navigation commands
        if any(word in cmd_lower for word in ['go to', 'move to', 'navigate to']):
            self.navigate_to_location(transcript)
        elif 'move forward' in cmd_lower or 'go forward' in cmd_lower:
            self.move_forward()
        elif 'move backward' in cmd_lower or 'go backward' in cmd_lower:
            self.move_backward()
        elif 'turn left' in cmd_lower:
            self.turn_left()
        elif 'turn right' in cmd_lower:
            self.turn_right()
        elif 'stop' in cmd_lower:
            self.stop_robot()
        # Manipulation commands
        elif any(word in cmd_lower for word in ['pick up', 'grasp', 'grab']):
            self.grasp_object()
        elif any(word in cmd_lower for word in ['place', 'put down', 'release']):
            self.release_object()
        # Query commands
        elif 'where are you' in cmd_lower:
            self.tell_location()
        elif 'battery' in cmd_lower:
            self.report_battery()

        # Publish command for other nodes
        cmd_msg = String()
        cmd_msg.data = transcript
        self.voice_cmd_pub.publish(cmd_msg)

    def navigate_to_location(self, command):
        """
        Navigate to a specific location.
        """
        # Extract location from command (simplified)
        # In practice, use NLP to extract location entities
        if 'kitchen' in command.lower():
            self.send_navigation_goal(-2.0, 1.0, 0.0)  # Example coordinates
        elif 'living room' in command.lower():
            self.send_navigation_goal(1.0, -1.0, 0.0)
        elif 'bedroom' in command.lower():
            self.send_navigation_goal(2.0, 2.0, 0.0)

    def move_forward(self):
        """Move robot forward."""
        cmd = Twist()
        cmd.linear.x = 0.3
        self.cmd_vel_pub.publish(cmd)

    def move_backward(self):
        """Move robot backward."""
        cmd = Twist()
        cmd.linear.x = -0.3
        self.cmd_vel_pub.publish(cmd)

    def turn_left(self):
        """Turn robot left."""
        cmd = Twist()
        cmd.angular.z = 0.5
        self.cmd_vel_pub.publish(cmd)

    def turn_right(self):
        """Turn robot right."""
        cmd = Twist()
        cmd.angular.z = -0.5
        self.cmd_vel_pub.publish(cmd)

    def stop_robot(self):
        """Stop robot movement."""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    def grasp_object(self):
        """Command robot to grasp an object."""
        self.get_logger().info('Grasping object')
        # Publish gripper command
        gripper_cmd = String()
        gripper_cmd.data = 'close'
        # self.gripper_pub.publish(gripper_cmd)  # Assuming gripper publisher exists

    def release_object(self):
        """Command robot to release an object."""
        self.get_logger().info('Releasing object')
        # Publish gripper command
        gripper_cmd = String()
        gripper_cmd.data = 'open'
        # self.gripper_pub.publish(gripper_cmd)  # Assuming gripper publisher exists

    def tell_location(self):
        """Report robot's current location."""
        self.get_logger().info(f'Current location: {self.robot_state["current_location"]}')

    def report_battery(self):
        """Report battery level."""
        self.get_logger().info(f'Battery level: {self.robot_state["battery_level"]}%')

    def send_navigation_goal(self, x, y, theta):
        """Send navigation goal to navigation system."""
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0

        # Convert theta to quaternion
        from math import sin, cos
        cy = cos(theta * 0.5)
        sy = sin(theta * 0.5)
        goal.pose.orientation.z = sy
        goal.pose.orientation.w = cy

        self.nav_goal_pub.publish(goal.pose)
        self.get_logger().info(f'Navigating to ({x}, {y})')


def main(args=None):
    rclpy.init(args=args)

    node = IntegratedWhisperRobot()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Best Practices for Whisper Integration

### Performance Considerations

1. **Model Selection**: Choose the appropriate model size based on your computational resources
2. **Audio Quality**: Use good quality microphones and audio preprocessing
3. **Real-time Processing**: Implement threading and queuing for non-blocking operation
4. **Error Handling**: Implement robust error handling for various failure modes
5. **Context Awareness**: Maintain conversation context for better command interpretation
6. **Security**: Consider privacy implications of voice data processing

Whisper integration with robotics opens up new possibilities for natural human-robot interaction, enabling robots to understand and respond to spoken commands in real-world environments.
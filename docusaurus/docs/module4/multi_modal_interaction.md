# Multi-Modal Interaction: Speech, Gesture, Vision

## Introduction

Multi-modal interaction in robotics combines multiple sensory modalities—speech, gesture, and vision—to create more natural and intuitive human-robot interfaces. This chapter explores how to integrate these modalities to enable robots to understand complex human intentions and respond appropriately. By combining multiple input channels, robots can achieve a more human-like understanding of their environment and user commands.

## Understanding Multi-Modal Interaction

### The Need for Multi-Modal Systems

Single-modality interfaces have limitations:
- **Speech alone**: Can be ambiguous without visual context
- **Vision alone**: May miss verbal instructions or preferences
- **Gesture alone**: Limited expressiveness without context

Multi-modal systems overcome these limitations by:
- **Disambiguation**: Resolving ambiguities across modalities
- **Context Enhancement**: Providing richer situational understanding
- **Natural Interaction**: Mirroring human communication patterns
- **Robustness**: Providing redundancy when one modality fails

### Multi-Modal Architecture

The multi-modal interaction architecture typically includes:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Speech        │    │   Vision        │    │   Gesture       │
│   Input         │    │   Input         │    │   Input         │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │ Speech      │      │ Vision      │      │ Gesture     │
    │ Processing  │      │ Processing  │      │ Processing  │
    └─────────────┘      └─────────────┘      └─────────────┘
          │                      │                      │
          └──────────┬───────────┼──────────────────────┘
                     ▼           ▼
            ┌─────────────────────────┐
            │   Multi-Modal Fusion    │
            │   and Intent Analysis   │
            └─────────────────────────┘
                           │
                           ▼
                   ┌─────────────┐
                   │   Action    │
                   │   Planning  │
                   └─────────────┘
                           │
                           ▼
                   ┌─────────────┐
                   │   Response  │
                   │   Generation│
                   └─────────────┘
```

## Speech Processing for Multi-Modal Systems

### Speech Recognition and Context Integration

```python
import speech_recognition as sr
import asyncio
import queue
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class SpeechEvent:
    """
    Data structure for speech events with timing and context.
    """
    text: str
    confidence: float
    timestamp: float
    speaker_id: Optional[str] = None
    context: Dict[str, Any] = None


class MultiModalSpeechProcessor:
    """
    Speech processor designed for multi-modal interaction.
    """

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # Speech event queue
        self.speech_queue = queue.Queue()

        # Parameters
        self.energy_threshold = 400  # Minimum audio energy to consider for recording
        self.pause_threshold = 0.8   # Seconds of non-speaking audio before a phrase is considered complete
        self.phrase_threshold = 0.3  # Minimum seconds of speaking audio before we consider the audio a phrase

        # Context information
        self.current_context = {}
        self.speaker_tracking = {}

    def start_listening(self):
        """
        Start listening for speech with callback.
        """
        self.stop_listening = False

        def audio_callback(recognizer, audio):
            if not self.stop_listening:
                try:
                    # Use Google Speech Recognition (alternatively, use offline models)
                    text = recognizer.recognize_google(audio)

                    # Create speech event with current context
                    speech_event = SpeechEvent(
                        text=text,
                        confidence=0.9,  # Would be obtained from recognizer if supported
                        timestamp=time.time(),
                        context=self.current_context.copy()
                    )

                    self.speech_queue.put(speech_event)

                except sr.UnknownValueError:
                    # Speech was detected but not understood
                    pass
                except sr.RequestError as e:
                    print(f"Could not request results from speech recognition service; {e}")

        # Start listening in background
        self.stopper = self.recognizer.listen_in_background(self.microphone, audio_callback)

        print("Listening for speech... Press Ctrl+C to stop.")

    def stop_listening(self):
        """
        Stop speech recognition.
        """
        if hasattr(self, 'stopper'):
            self.stopper()
        self.stop_listening = True

    def get_speech_events(self):
        """
        Get speech events from queue.
        """
        events = []
        try:
            while True:
                event = self.speech_queue.get_nowait()
                events.append(event)
        except queue.Empty:
            pass
        return events

    def update_context(self, context_updates: Dict[str, Any]):
        """
        Update context information for speech processing.
        """
        self.current_context.update(context_updates)


class AdvancedSpeechProcessor(MultiModalSpeechProcessor):
    """
    Advanced speech processor with speaker diarization and emotion detection.
    """

    def __init__(self):
        super().__init__()

        # Initialize additional components
        self.speaker_model = self.load_speaker_model()
        self.emotion_model = self.load_emotion_model()
        self.language_model = self.load_language_model()

    def load_speaker_model(self):
        """
        Load speaker identification model.
        """
        # In practice, this would load a pre-trained speaker identification model
        # For example, using scikit-learn, pyannote.audio, or similar
        return None  # Placeholder

    def load_emotion_model(self):
        """
        Load emotion detection model.
        """
        # In practice, this would load a pre-trained emotion detection model
        # For example, using librosa for audio features + sklearn/pytorch model
        return None  # Placeholder

    def load_language_model(self):
        """
        Load language model for better understanding.
        """
        # This could be a transformer-based model for context understanding
        return None  # Placeholder

    def process_speech_with_context(self, audio_data, visual_context=None, gesture_context=None):
        """
        Process speech with additional context from other modalities.
        """
        try:
            # Recognize speech
            text = self.recognizer.recognize_google(audio_data)

            # Analyze speaker (if multiple speakers)
            speaker_id = self.identify_speaker(audio_data)

            # Detect emotion from speech
            emotion = self.detect_emotion(audio_data)

            # Analyze with language model using context
            analyzed_text = self.analyze_with_context(
                text,
                visual_context=visual_context,
                gesture_context=gesture_context
            )

            return SpeechEvent(
                text=analyzed_text,
                confidence=0.9,
                timestamp=time.time(),
                speaker_id=speaker_id,
                context={
                    'emotion': emotion,
                    'visual_context': visual_context,
                    'gesture_context': gesture_context
                }
            )

        except Exception as e:
            print(f"Error processing speech: {e}")
            return None

    def identify_speaker(self, audio_data):
        """
        Identify speaker from audio data.
        """
        # Implementation would use speaker identification model
        return "unknown_speaker"  # Placeholder

    def detect_emotion(self, audio_data):
        """
        Detect emotion from audio data.
        """
        # Implementation would use emotion detection model
        return "neutral"  # Placeholder

    def analyze_with_context(self, text, visual_context=None, gesture_context=None):
        """
        Analyze text with additional context from other modalities.
        """
        # This would integrate visual and gesture context to disambiguate speech
        # For example: "Pick that up" with visual context pointing to specific object
        return text  # Placeholder
```

## Vision Processing for Multi-Modal Systems

### Object Detection and Attention

```python
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class DetectedObject:
    """
    Data structure for detected objects.
    """
    id: str
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]  # x, y center
    color: Optional[str] = None
    size: Optional[str] = None


@dataclass
class VisualEvent:
    """
    Data structure for visual events.
    """
    objects: List[DetectedObject]
    attention_regions: List[Tuple[int, int, int, int]]  # x1, y1, x2, y2
    gaze_direction: Optional[Tuple[float, float]] = None  # x, y direction vector
    timestamp: float = 0.0


class MultiModalVisionProcessor:
    """
    Vision processor designed for multi-modal interaction.
    """

    def __init__(self):
        # Initialize models
        self.object_detector = self.load_object_detector()
        self.pose_estimator = self.load_pose_estimator()
        self.gaze_estimator = self.load_gaze_estimator()

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Processing parameters
        self.detection_confidence = 0.5
        self.max_objects = 10

        # Attention tracking
        self.attention_history = []
        self.focus_objects = set()

    def load_object_detector(self):
        """
        Load object detection model (e.g., YOLO, SSD, etc.).
        """
        # This would load a pre-trained object detection model
        # For example, using torchvision.models or YOLOv5
        try:
            import torchvision.models.detection as det_models
            model = det_models.fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()
            return model
        except ImportError:
            print("Torchvision not available, using placeholder")
            return None

    def load_pose_estimator(self):
        """
        Load human pose estimation model.
        """
        # This would load a pose estimation model like OpenPose or MediaPipe
        return None  # Placeholder

    def load_gaze_estimator(self):
        """
        Load gaze estimation model.
        """
        # This would load a gaze estimation model
        return None  # Placeholder

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the camera.
        """
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def detect_objects(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects in the frame.
        """
        if self.object_detector is None:
            # Placeholder implementation
            return self.placeholder_object_detection(frame)

        # Convert frame to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Preprocess image
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        input_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = self.object_detector(input_tensor)

        # Process outputs
        objects = []
        if len(outputs) > 0:
            output = outputs[0]
            boxes = output['boxes'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            scores = output['scores'].cpu().numpy()

            for i in range(len(scores)):
                if scores[i] > self.detection_confidence:
                    bbox = boxes[i].astype(int)
                    class_name = self.coco_class_names.get(labels[i], f'unknown_{labels[i]}')

                    obj = DetectedObject(
                        id=f'obj_{i}',
                        class_name=class_name,
                        confidence=float(scores[i]),
                        bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                        center=((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    )
                    objects.append(obj)

                    # Limit number of objects
                    if len(objects) >= self.max_objects:
                        break

        return objects

    def placeholder_object_detection(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Placeholder object detection implementation.
        """
        # This is a simple color-based detection for demonstration
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color ranges for common objects
        color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255]),
        }

        objects = []
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x, center_y = x + w // 2, y + h // 2

                    obj = DetectedObject(
                        id=f'{color_name}_{len(objects)}',
                        class_name=color_name,
                        confidence=0.8,
                        bbox=(x, y, x + w, y + h),
                        center=(center_x, center_y),
                        color=color_name
                    )
                    objects.append(obj)

        return objects

    @property
    def coco_class_names(self):
        """
        COCO dataset class names.
        """
        return {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
            16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
            22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
            28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
            35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
            40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
            44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
            51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
            56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
            61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
            67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
            75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
            80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
            86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
        }

    def estimate_human_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Estimate human pose in the frame.
        """
        # This would use a pose estimation model
        # For demonstration, return placeholder
        return None

    def estimate_gaze_direction(self, frame: np.ndarray, face_bbox: Tuple) -> Optional[Tuple[float, float]]:
        """
        Estimate gaze direction from face region.
        """
        # This would use a gaze estimation model
        # For demonstration, return placeholder
        return None

    def process_frame(self) -> Optional[VisualEvent]:
        """
        Process a single frame to extract visual information.
        """
        frame = self.capture_frame()
        if frame is None:
            return None

        # Detect objects
        objects = self.detect_objects(frame)

        # Estimate human pose (if people detected)
        pose_info = self.estimate_human_pose(frame)

        # Estimate gaze direction (if faces detected)
        gaze_direction = None
        if pose_info and 'face' in pose_info:
            gaze_direction = self.estimate_gaze_direction(frame, pose_info['face'])

        # Identify attention regions (based on detected objects and human pose)
        attention_regions = self.identify_attention_regions(objects, pose_info)

        # Create visual event
        visual_event = VisualEvent(
            objects=objects,
            attention_regions=attention_regions,
            gaze_direction=gaze_direction,
            timestamp=time.time()
        )

        # Update attention history
        self.attention_history.append(visual_event)

        # Keep only recent history
        if len(self.attention_history) > 10:
            self.attention_history = self.attention_history[-10:]

        return visual_event

    def identify_attention_regions(self, objects: List[DetectedObject], pose_info: Optional[Dict]) -> List[Tuple[int, int, int, int]]:
        """
        Identify regions of visual attention based on objects and human pose.
        """
        attention_regions = []

        # If human is detected and looking at specific object
        if pose_info and 'gaze_target' in pose_info:
            # Add region around gaze target
            target_region = pose_info['gaze_target']
            attention_regions.append(target_region)

        # Add regions around detected objects that are likely to be attended
        for obj in objects:
            # Consider size, position, and other factors
            if obj.confidence > 0.7:  # High confidence detection
                attention_regions.append(obj.bbox)

        return attention_regions

    def get_attention_context(self) -> Dict:
        """
        Get current attention context for multi-modal fusion.
        """
        if not self.attention_history:
            return {}

        latest_event = self.attention_history[-1]

        context = {
            'visible_objects': [obj.class_name for obj in latest_event.objects],
            'focused_object': self.get_focused_object(latest_event),
            'attention_regions': latest_event.attention_regions,
            'gaze_direction': latest_event.gaze_direction
        }

        return context

    def get_focused_object(self, visual_event: VisualEvent) -> Optional[DetectedObject]:
        """
        Determine which object is currently being focused on.
        """
        # This would use attention models, gaze estimation, etc.
        # For now, return the closest object
        if visual_event.objects:
            # Sort by y-coordinate (assuming closer objects are lower in frame)
            closest_obj = min(visual_event.objects, key=lambda obj: obj.center[1])
            return closest_obj

        return None
```

## Gesture Recognition and Processing

### Hand and Body Gesture Recognition

```python
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Gesture:
    """
    Data structure for recognized gestures.
    """
    name: str
    confidence: float
    landmarks: List[Tuple[float, float, float]]  # x, y, z coordinates
    gesture_type: str  # 'hand', 'body', 'head', etc.
    timestamp: float = 0.0


@dataclass
class GestureEvent:
    """
    Data structure for gesture events.
    """
    gesture: Gesture
    context: Dict = None
    timestamp: float = 0.0


class MultiModalGestureProcessor:
    """
    Gesture processor designed for multi-modal interaction.
    """

    def __init__(self):
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize gesture recognizers
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Gesture vocabulary
        self.gesture_vocabulary = self.define_gesture_vocabulary()

        # Processing parameters
        self.min_gesture_confidence = 0.6
        self.gesture_history = []

    def define_gesture_vocabulary(self) -> Dict[str, Dict]:
        """
        Define the vocabulary of recognizable gestures.
        """
        return {
            'pointing': {
                'description': 'Index finger extended, other fingers folded',
                'key_landmarks': [8],  # Tip of index finger
                'constraints': {
                    'index_finger_extended': True,
                    'other_fingers_folded': True
                }
            },
            'thumbs_up': {
                'description': 'Thumb up, other fingers folded',
                'key_landmarks': [4, 8],  # Thumb tip and index finger MCP
                'constraints': {
                    'thumb_extended': True,
                    'other_fingers_folded': True
                }
            },
            'wave': {
                'description': 'Hand waving motion',
                'key_landmarks': [8, 12],  # Index and middle finger tips
                'constraints': {
                    'periodic_motion': True
                }
            },
            'okay': {
                'description': 'OK hand sign (thumb and index finger touching)',
                'key_landmarks': [4, 8],  # Thumb tip and index finger tip
                'constraints': {
                    'thumb_index_touching': True
                }
            },
            'stop': {
                'description': 'Palm facing forward (stop sign)',
                'key_landmarks': [8, 12, 16, 20],  # All finger tips
                'constraints': {
                    'fingers_extended': True,
                    'palm_facing_forward': True
                }
            }
        }

    def recognize_hand_gestures(self, frame: np.ndarray) -> List[Gesture]:
        """
        Recognize hand gestures from frame.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        gestures = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y, landmark.z))

                # Recognize gesture
                gesture_name, confidence = self.classify_gesture(landmarks, 'hand')

                if confidence > self.min_gesture_confidence:
                    gesture = Gesture(
                        name=gesture_name,
                        confidence=confidence,
                        landmarks=landmarks,
                        gesture_type='hand',
                        timestamp=time.time()
                    )
                    gestures.append(gesture)

                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

        return gestures

    def classify_gesture(self, landmarks: List[Tuple[float, float, float]], gesture_type: str) -> Tuple[str, float]:
        """
        Classify gesture based on landmarks.
        """
        if gesture_type == 'hand':
            # Analyze finger positions and relationships
            return self.classify_hand_gesture(landmarks)

        return 'unknown', 0.0

    def classify_hand_gesture(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[str, float]:
        """
        Classify hand gesture based on landmark positions.
        """
        # Calculate distances between key points
        if len(landmarks) < 21:  # Not enough landmarks
            return 'unknown', 0.0

        # Extract specific landmark positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]

        # Calculate distances
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
        index_middle_dist = self.calculate_distance(index_tip, middle_tip)
        middle_ring_dist = self.calculate_distance(middle_tip, ring_tip)
        ring_pinky_dist = self.calculate_distance(ring_tip, pinky_tip)

        # Classify based on distances and positions
        if self.is_pointing_gesture(landmarks):
            return 'pointing', 0.9
        elif self.is_thumbs_up_gesture(landmarks):
            return 'thumbs_up', 0.85
        elif self.is_okay_gesture(landmarks):
            return 'okay', 0.8
        elif self.is_stop_gesture(landmarks):
            return 'stop', 0.85
        else:
            return 'unknown', 0.0

    def is_pointing_gesture(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        Check if landmarks represent a pointing gesture.
        """
        # Index finger extended, others folded
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]

        # Index finger should be extended (tip further from MCP than others)
        index_extended = self.calculate_distance(index_tip, index_mcp) > 0.1
        others_folded = (
            self.calculate_distance(middle_tip, middle_mcp) < 0.08 and
            self.calculate_distance(ring_tip, ring_mcp) < 0.08 and
            self.calculate_distance(pinky_tip, pinky_mcp) < 0.08
        )

        return index_extended and others_folded

    def is_thumbs_up_gesture(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        Check if landmarks represent a thumbs up gesture.
        """
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]

        # Thumb extended, others folded
        thumb_extended = self.calculate_distance(thumb_tip, thumb_mcp) > 0.1
        others_folded = (
            self.calculate_distance(index_tip, index_mcp) < 0.08 and
            self.calculate_distance(middle_tip, middle_mcp) < 0.08 and
            self.calculate_distance(ring_tip, ring_mcp) < 0.08 and
            self.calculate_distance(pinky_tip, pinky_mcp) < 0.08
        )

        return thumb_extended and others_folded

    def is_okay_gesture(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        Check if landmarks represent an OK gesture.
        """
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        # Thumb tip and index finger tip close together
        distance = self.calculate_distance(thumb_tip, index_tip)
        return distance < 0.05

    def is_stop_gesture(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        Check if landmarks represent a stop gesture (open palm).
        """
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]

        # All fingers extended
        all_extended = (
            self.calculate_distance(index_tip, index_mcp) > 0.1 and
            self.calculate_distance(middle_tip, middle_mcp) > 0.1 and
            self.calculate_distance(ring_tip, ring_mcp) > 0.1 and
            self.calculate_distance(pinky_tip, pinky_mcp) > 0.1
        )

        return all_extended

    def calculate_distance(self, point1: Tuple[float, float, float], point2: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two 3D points.
        """
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

    def recognize_body_gestures(self, frame: np.ndarray) -> List[Gesture]:
        """
        Recognize body gestures from frame.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        gestures = []

        if results.pose_landmarks:
            # Extract body landmarks
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))

            # Recognize body gesture (simplified)
            gesture_name, confidence = self.classify_body_gesture(landmarks)

            if confidence > self.min_gesture_confidence:
                gesture = Gesture(
                    name=gesture_name,
                    confidence=confidence,
                    landmarks=landmarks,
                    gesture_type='body',
                    timestamp=time.time()
                )
                gestures.append(gesture)

            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

        return gestures

    def classify_body_gesture(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[str, float]:
        """
        Classify body gesture based on landmarks.
        """
        # Simplified body gesture classification
        # In practice, this would use more sophisticated analysis

        # Example: Raise hand gesture
        nose = landmarks[0]  # Nose landmark
        left_wrist = landmarks[15]  # Left wrist
        right_wrist = landmarks[16]  # Right wrist

        # Check if either hand is raised above head
        if left_wrist[1] < nose[1] or right_wrist[1] < nose[1]:  # Y-axis is inverted in MediaPipe
            return 'raise_hand', 0.8

        return 'unknown', 0.0

    def process_frame(self, frame: np.ndarray) -> List[GestureEvent]:
        """
        Process frame to extract gestures.
        """
        # Recognize hand gestures
        hand_gestures = self.recognize_hand_gestures(frame)

        # Recognize body gestures
        body_gestures = self.recognize_body_gestures(frame)

        # Combine all gestures
        all_gestures = hand_gestures + body_gestures

        # Create gesture events
        gesture_events = []
        for gesture in all_gestures:
            event = GestureEvent(
                gesture=gesture,
                timestamp=time.time()
            )
            gesture_events.append(event)

            # Update gesture history
            self.gesture_history.append(event)

        # Keep only recent history
        if len(self.gesture_history) > 20:
            self.gesture_history = self.gesture_history[-20:]

        return gesture_events

    def get_gesture_context(self) -> Dict:
        """
        Get current gesture context for multi-modal fusion.
        """
        if not self.gesture_history:
            return {}

        recent_gestures = self.gesture_history[-5:]  # Last 5 gestures

        context = {
            'recent_gestures': [(g.gesture.name, g.gesture.confidence) for g in recent_gestures],
            'dominant_gesture': self.get_dominant_gesture(recent_gestures),
            'gesture_sequence': [g.gesture.name for g in recent_gestures]
        }

        return context

    def get_dominant_gesture(self, recent_gestures: List[GestureEvent]) -> Optional[str]:
        """
        Determine the dominant gesture from recent history.
        """
        if not recent_gestures:
            return None

        # Group by gesture name and find most confident
        gesture_counts = {}
        gesture_confidences = {}

        for event in recent_gestures:
            name = event.gesture.name
            if name not in gesture_counts:
                gesture_counts[name] = 0
                gesture_confidences[name] = 0.0

            gesture_counts[name] += 1
            gesture_confidences[name] = max(gesture_confidences[name], event.gesture.confidence)

        # Find gesture with highest confidence
        dominant_gesture = max(gesture_confidences.keys(), key=lambda x: gesture_confidences[x])
        return dominant_gesture
```

## Multi-Modal Fusion and Intent Recognition

### Fusion Engine

```python
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MultiModalEvent:
    """
    Combined event from multiple modalities.
    """
    speech_event: Optional[SpeechEvent] = None
    visual_event: Optional[VisualEvent] = None
    gesture_event: Optional[GestureEvent] = None
    timestamp: float = 0.0
    fused_intent: Optional[str] = None
    confidence: float = 0.0


class MultiModalFusionEngine:
    """
    Fuses information from multiple modalities to recognize intent.
    """

    def __init__(self):
        # Queues for different modalities
        self.speech_queue = queue.Queue()
        self.visual_queue = queue.Queue()
        self.gesture_queue = queue.Queue()

        # Event synchronization
        self.speech_event = None
        self.visual_event = None
        self.gesture_event = None

        # Fusion parameters
        self.temporal_window = 2.0  # seconds to consider for fusion
        self.confidence_weights = {
            'speech': 0.5,
            'visual': 0.3,
            'gesture': 0.2
        }

        # Intent vocabulary
        self.intent_vocabulary = self.define_intent_vocabulary()

    def define_intent_vocabulary(self) -> Dict[str, Dict]:
        """
        Define the vocabulary of possible intents.
        """
        return {
            'navigation': {
                'keywords': ['go', 'move', 'navigate', 'to', 'toward'],
                'gestures': ['pointing'],
                'visual_context': ['location', 'waypoint']
            },
            'manipulation': {
                'keywords': ['pick', 'grasp', 'take', 'lift', 'put', 'place', 'drop'],
                'gestures': ['pointing', 'grasping_motion'],
                'visual_context': ['object', 'graspable']
            },
            'detection': {
                'keywords': ['find', 'locate', 'look', 'see', 'where', 'is'],
                'gestures': ['pointing', 'searching_motion'],
                'visual_context': ['object', 'search_area']
            },
            'greeting': {
                'keywords': ['hello', 'hi', 'hey', 'good morning', 'good evening'],
                'gestures': ['wave', 'nod'],
                'visual_context': ['person', 'face']
            },
            'affirmation': {
                'keywords': ['yes', 'yeah', 'sure', 'ok', 'okay', 'alright'],
                'gestures': ['thumbs_up', 'nod'],
                'visual_context': ['interaction']
            },
            'negation': {
                'keywords': ['no', 'nope', 'negative', 'stop', 'cancel'],
                'gestures': ['stop', 'shake_head'],
                'visual_context': ['ongoing_action']
            }
        }

    def add_speech_event(self, event: SpeechEvent):
        """
        Add speech event to fusion engine.
        """
        self.speech_queue.put(event)
        self.speech_event = event

    def add_visual_event(self, event: VisualEvent):
        """
        Add visual event to fusion engine.
        """
        self.visual_queue.put(event)
        self.visual_event = event

    def add_gesture_event(self, event: GestureEvent):
        """
        Add gesture event to fusion engine.
        """
        self.gesture_queue.put(event)
        self.gesture_event = event

    def fuse_modalities(self) -> Optional[MultiModalEvent]:
        """
        Fuse information from all modalities to recognize intent.
        """
        # Get recent events within temporal window
        speech_event = self.get_recent_speech_event()
        visual_event = self.get_recent_visual_event()
        gesture_event = self.get_recent_gesture_event()

        if not any([speech_event, visual_event, gesture_event]):
            return None

        # Perform fusion
        fused_intent, confidence = self.perform_fusion(
            speech_event, visual_event, gesture_event
        )

        # Create multi-modal event
        mm_event = MultiModalEvent(
            speech_event=speech_event,
            visual_event=visual_event,
            gesture_event=gesture_event,
            timestamp=time.time(),
            fused_intent=fused_intent,
            confidence=confidence
        )

        return mm_event

    def get_recent_speech_event(self) -> Optional[SpeechEvent]:
        """
        Get speech event within temporal window.
        """
        recent_events = []
        current_time = time.time()

        # Get all events from queue
        while not self.speech_queue.empty():
            try:
                event = self.speech_queue.get_nowait()
                if current_time - event.timestamp <= self.temporal_window:
                    recent_events.append(event)
            except queue.Empty:
                break

        # Return most recent or combine if needed
        if recent_events:
            return recent_events[-1]

        return self.speech_event if self.speech_event else None

    def get_recent_visual_event(self) -> Optional[VisualEvent]:
        """
        Get visual event within temporal window.
        """
        recent_events = []
        current_time = time.time()

        # Get all events from queue
        while not self.visual_queue.empty():
            try:
                event = self.visual_queue.get_nowait()
                if current_time - event.timestamp <= self.temporal_window:
                    recent_events.append(event)
            except queue.Empty:
                break

        # Return most recent
        if recent_events:
            return recent_events[-1]

        return self.visual_event if self.visual_event else None

    def get_recent_gesture_event(self) -> Optional[GestureEvent]:
        """
        Get gesture event within temporal window.
        """
        recent_events = []
        current_time = time.time()

        # Get all events from queue
        while not self.gesture_queue.empty():
            try:
                event = self.gesture_queue.get_nowait()
                if current_time - event.timestamp <= self.temporal_window:
                    recent_events.append(event)
            except queue.Empty:
                break

        # Return most recent
        if recent_events:
            return recent_events[-1]

        return self.gesture_event if self.gesture_event else None

    def perform_fusion(self, speech_event: Optional[SpeechEvent],
                      visual_event: Optional[VisualEvent],
                      gesture_event: Optional[GestureEvent]) -> Tuple[str, float]:
        """
        Perform multi-modal fusion to recognize intent.
        """
        # Extract information from each modality
        speech_info = self.extract_speech_info(speech_event)
        visual_info = self.extract_visual_info(visual_event)
        gesture_info = self.extract_gesture_info(gesture_event)

        # Calculate confidence scores for each intent
        intent_scores = {}

        for intent, definition in self.intent_vocabulary.items():
            score = 0.0
            total_weight = 0.0

            # Speech contribution
            if speech_info:
                speech_match = self.match_keywords(speech_info, definition['keywords'])
                score += speech_match * self.confidence_weights['speech']
                total_weight += self.confidence_weights['speech']

            # Visual contribution
            if visual_info:
                visual_match = self.match_visual_context(visual_info, definition['visual_context'])
                score += visual_match * self.confidence_weights['visual']
                total_weight += self.confidence_weights['visual']

            # Gesture contribution
            if gesture_info:
                gesture_match = self.match_gestures(gesture_info, definition['gestures'])
                score += gesture_match * self.confidence_weights['gesture']
                total_weight += self.confidence_weights['gesture']

            # Normalize by total weight
            if total_weight > 0:
                intent_scores[intent] = score / total_weight
            else:
                intent_scores[intent] = 0.0

        # Find intent with highest score
        if intent_scores:
            best_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x])
            best_confidence = intent_scores[best_intent]
            return best_intent, best_confidence

        return 'unknown', 0.0

    def extract_speech_info(self, speech_event: Optional[SpeechEvent]) -> Dict[str, Any]:
        """
        Extract relevant information from speech event.
        """
        if not speech_event:
            return {}

        text = speech_event.text.lower()
        words = text.split()

        return {
            'text': text,
            'words': words,
            'original_text': speech_event.text,
            'confidence': speech_event.confidence
        }

    def extract_visual_info(self, visual_event: Optional[VisualEvent]) -> Dict[str, Any]:
        """
        Extract relevant information from visual event.
        """
        if not visual_event:
            return {}

        objects = [obj.class_name for obj in visual_event.objects]
        object_names = list(set(objects))  # Unique object names

        return {
            'objects': object_names,
            'object_count': len(visual_event.objects),
            'attention_regions': visual_event.attention_regions,
            'gaze_direction': visual_event.gaze_direction
        }

    def extract_gesture_info(self, gesture_event: Optional[GestureEvent]) -> Dict[str, Any]:
        """
        Extract relevant information from gesture event.
        """
        if not gesture_event:
            return {}

        return {
            'gesture_name': gesture_event.gesture.name,
            'gesture_confidence': gesture_event.gesture.confidence,
            'gesture_type': gesture_event.gesture.gesture_type
        }

    def match_keywords(self, speech_info: Dict, keywords: List[str]) -> float:
        """
        Match speech against keywords to determine relevance.
        """
        if not speech_info or not keywords:
            return 0.0

        text = speech_info['text']
        confidence = speech_info.get('confidence', 1.0)

        matches = 0
        total_keywords = len(keywords)

        for keyword in keywords:
            if keyword in text:
                matches += 1

        match_ratio = matches / total_keywords if total_keywords > 0 else 0.0
        return match_ratio * confidence

    def match_visual_context(self, visual_info: Dict, context_keywords: List[str]) -> float:
        """
        Match visual information against context keywords.
        """
        if not visual_info or not context_keywords:
            return 0.0

        objects = visual_info.get('objects', [])
        attention_regions = visual_info.get('attention_regions', [])

        matches = 0
        total_keywords = len(context_keywords)

        for keyword in context_keywords:
            # Check if keyword matches any detected objects
            for obj in objects:
                if keyword in obj.lower():
                    matches += 1
                    break

            # Check if keyword relates to context
            if keyword in ['location', 'waypoint'] and attention_regions:
                matches += 1

        match_ratio = matches / total_keywords if total_keywords > 0 else 0.0
        return match_ratio

    def match_gestures(self, gesture_info: Dict, allowed_gestures: List[str]) -> float:
        """
        Match gesture against allowed gestures.
        """
        if not gesture_info or not allowed_gestures:
            return 0.0

        gesture_name = gesture_info.get('gesture_name', '')
        gesture_confidence = gesture_info.get('gesture_confidence', 1.0)

        if gesture_name in allowed_gestures:
            return gesture_confidence

        return 0.0


class MultiModalInteractionManager:
    """
    Main manager for multi-modal interaction system.
    """

    def __init__(self):
        # Initialize processors
        self.speech_processor = AdvancedSpeechProcessor()
        self.vision_processor = MultiModalVisionProcessor()
        self.gesture_processor = MultiModalGestureProcessor()
        self.fusion_engine = MultiModalFusionEngine()

        # Threading for concurrent processing
        self.running = False
        self.speech_thread = None
        self.vision_thread = None
        self.gesture_thread = None

        # Callback for fused intents
        self.intent_callback = None

    def start_interaction(self):
        """
        Start multi-modal interaction system.
        """
        self.running = True

        # Start speech processing
        self.speech_thread = threading.Thread(target=self.speech_processing_loop, daemon=True)
        self.speech_thread.start()

        # Start vision processing
        self.vision_thread = threading.Thread(target=self.vision_processing_loop, daemon=True)
        self.vision_thread.start()

        # Start gesture processing
        self.gesture_thread = threading.Thread(target=self.gesture_processing_loop, daemon=True)
        self.gesture_thread.start()

        # Start fusion processing
        self.fusion_thread = threading.Thread(target=self.fusion_processing_loop, daemon=True)
        self.fusion_thread.start()

        print("Multi-modal interaction system started")

    def stop_interaction(self):
        """
        Stop multi-modal interaction system.
        """
        self.running = False
        print("Stopping multi-modal interaction system...")

    def set_intent_callback(self, callback):
        """
        Set callback for when fused intents are recognized.
        """
        self.intent_callback = callback

    def speech_processing_loop(self):
        """
        Continuously process speech input.
        """
        self.speech_processor.start_listening()

        while self.running:
            speech_events = self.speech_processor.get_speech_events()
            for event in speech_events:
                self.fusion_engine.add_speech_event(event)
                time.sleep(0.01)  # Small delay to prevent busy waiting

        self.speech_processor.stop_listening()

    def vision_processing_loop(self):
        """
        Continuously process visual input.
        """
        while self.running:
            visual_event = self.vision_processor.process_frame()
            if visual_event:
                self.fusion_engine.add_visual_event(visual_event)

            time.sleep(0.1)  # ~10 FPS processing

    def gesture_processing_loop(self):
        """
        Continuously process gesture input.
        """
        cap = cv2.VideoCapture(0)

        while self.running:
            ret, frame = cap.read()
            if ret:
                gesture_events = self.gesture_processor.process_frame(frame)
                for event in gesture_events:
                    self.fusion_engine.add_gesture_event(event)

            time.sleep(0.05)  # ~20 FPS processing

        cap.release()

    def fusion_processing_loop(self):
        """
        Continuously perform multi-modal fusion.
        """
        while self.running:
            mm_event = self.fusion_engine.fuse_modalities()
            if mm_event and mm_event.fused_intent:
                print(f"Fused intent: {mm_event.fused_intent} (confidence: {mm_event.confidence:.2f})")

                # Call intent callback if set
                if self.intent_callback:
                    self.intent_callback(mm_event.fused_intent, mm_event.confidence, mm_event)

            time.sleep(0.05)  # ~20 fusion cycles per second

    def get_current_context(self) -> Dict[str, Any]:
        """
        Get current multi-modal context.
        """
        return {
            'speech_context': getattr(self.speech_processor, 'current_context', {}),
            'visual_context': self.vision_processor.get_attention_context(),
            'gesture_context': self.gesture_processor.get_gesture_context()
        }


def intent_handler(intent: str, confidence: float, mm_event: MultiModalEvent):
    """
    Example intent handler function.
    """
    print(f"Handling intent: {intent} with confidence {confidence}")

    # In a real system, this would translate the intent to robot actions
    if intent == 'navigation':
        print("Robot should navigate to specified location")
    elif intent == 'manipulation':
        print("Robot should manipulate specified object")
    elif intent == 'detection':
        print("Robot should detect specified object")
    elif intent == 'greeting':
        print("Robot should greet the person")
    elif intent == 'affirmation':
        print("Robot should acknowledge positively")
    elif intent == 'negation':
        print("Robot should acknowledge negatively")


def main():
    """
    Main function to demonstrate multi-modal interaction.
    """
    manager = MultiModalInteractionManager()

    # Set intent handler
    manager.set_intent_callback(intent_handler)

    print("Starting multi-modal interaction system...")
    print("Speak, gesture, or show objects to the camera to interact.")
    print("Press Ctrl+C to stop.")

    try:
        manager.start_interaction()

        # Keep running until interrupted
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
        manager.stop_interaction()


if __name__ == '__main__':
    main()
```

## Integration with ROS 2

### ROS 2 Multi-Modal Interface

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from multi_modal_msgs.msg import MultiModalEvent, IntentRecognition  # Custom message types


class MultiModalROSInterface(Node):
    """
    ROS 2 interface for multi-modal interaction system.
    """

    def __init__(self):
        super().__init__('multi_modal_interface')

        # Publishers
        self.intent_pub = self.create_publisher(IntentRecognition, '/multi_modal/intent', 10)
        self.event_pub = self.create_publisher(MultiModalEvent, '/multi_modal/event', 10)
        self.feedback_pub = self.create_publisher(String, '/multi_modal/feedback', 10)

        # Subscribers
        self.speech_sub = self.create_subscription(
            String,
            '/speech_recognition/text',
            self.speech_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Initialize multi-modal manager
        self.mm_manager = MultiModalInteractionManager()
        self.mm_manager.set_intent_callback(self.ros_intent_handler)

        # Start multi-modal system
        self.mm_manager_thread = threading.Thread(target=self.mm_manager.start_interaction, daemon=True)
        self.mm_manager_thread.start()

        self.get_logger().info('Multi-Modal ROS Interface initialized')

    def speech_callback(self, msg):
        """
        Handle speech input from ROS 2 topic.
        """
        # Convert ROS String to SpeechEvent and add to fusion engine
        speech_event = SpeechEvent(
            text=msg.data,
            confidence=0.9,  # Would come from speech recognition confidence
            timestamp=self.get_clock().now().nanoseconds / 1e9
        )
        self.mm_manager.fusion_engine.add_speech_event(speech_event)

    def image_callback(self, msg):
        """
        Handle image input from ROS 2 topic.
        """
        # Convert ROS Image to OpenCV format
        # This would require cv_bridge in real implementation
        pass

    def ros_intent_handler(self, intent: str, confidence: float, mm_event: MultiModalEvent):
        """
        Handle fused intents in ROS 2 context.
        """
        # Publish intent recognition
        intent_msg = IntentRecognition()
        intent_msg.intent = intent
        intent_msg.confidence = confidence
        intent_msg.timestamp = self.get_clock().now().to_msg()

        self.intent_pub.publish(intent_msg)

        # Publish detailed multi-modal event
        event_msg = MultiModalEvent()
        event_msg.speech_text = mm_event.speech_event.text if mm_event.speech_event else ""
        event_msg.visual_objects = [obj.class_name for obj in mm_event.visual_event.objects] if mm_event.visual_event else []
        event_msg.gesture_name = mm_event.gesture_event.gesture.name if mm_event.gesture_event else ""
        event_msg.fused_intent = mm_event.fused_intent
        event_msg.confidence = mm_event.confidence
        event_msg.header.stamp = self.get_clock().now().to_msg()

        self.event_pub.publish(event_msg)

        self.get_logger().info(f'Fused intent: {intent} (confidence: {confidence:.2f})')

    def publish_feedback(self, feedback_text: str):
        """
        Publish feedback to user.
        """
        feedback_msg = String()
        feedback_msg.data = feedback_text
        self.feedback_pub.publish(feedback_msg)


def main(args=None):
    rclpy.init(args=args)

    node = MultiModalROSInterface()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down multi-modal interface...')
    finally:
        node.mm_manager.stop_interaction()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Best Practices for Multi-Modal Systems

### Design Principles

1. **Temporal Synchronization**: Align events from different modalities in time
2. **Confidence Integration**: Combine confidences from different modalities appropriately
3. **Context Awareness**: Use context to disambiguate inputs
4. **Robustness**: Handle failure of individual modalities gracefully
5. **Latency Management**: Optimize processing for real-time interaction
6. **User Feedback**: Provide clear feedback about system understanding
7. **Privacy Considerations**: Respect privacy when processing audio/video
8. **Calibration**: Calibrate sensors and models regularly

Multi-modal interaction systems enable robots to understand human intentions through natural communication channels, creating more intuitive and effective human-robot collaboration. By combining speech, gesture, and vision, robots can achieve a more human-like understanding of their environment and user commands.
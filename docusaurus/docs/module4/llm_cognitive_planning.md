# Cognitive Planning using Large Language Models (LLMs)

## Introduction

Large Language Models (LLMs) have revolutionized how we approach cognitive planning in robotics. Unlike traditional symbolic planners that rely on predefined rules and state transitions, LLMs can understand natural language, reason about complex tasks, and generate sophisticated action sequences. This chapter explores how to integrate LLMs into robotic systems for cognitive planning, enabling robots to perform complex, multi-step tasks with human-like reasoning.

## Understanding Cognitive Planning in Robotics

### Traditional vs. LLM-Based Planning

Traditional planning in robotics involves:
- **Symbolic Representation**: States and actions represented symbolically
- **Predefined Rules**: Explicit rules for action selection
- **Deterministic Execution**: Predictable action sequences
- **Limited Flexibility**: Difficulty handling novel situations

LLM-based planning offers:
- **Natural Language Interface**: Direct command understanding
- **Contextual Reasoning**: Ability to reason about context
- **Adaptive Behavior**: Flexibility in handling novel situations
- **Knowledge Integration**: Incorporation of world knowledge

### Cognitive Planning Architecture

The cognitive planning architecture with LLMs typically includes:

```
Natural Language Input
        ↓
    LLM Parser (Intent Recognition)
        ↓
    Task Decomposition
        ↓
    Action Sequence Generation
        ↓
    ROS 2 Action Execution
        ↓
    Feedback and Monitoring
        ↓
    Plan Adjustment (if needed)
```

## Integrating LLMs with ROS 2

### LLM Communication Architecture

```python
import openai
import asyncio
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
import time


class LLMCognitivePlanner(Node):
    """
    Cognitive planning node using Large Language Models.
    """

    def __init__(self):
        super().__init__('llm_cognitive_planner')

        # Initialize LLM client
        self.llm_client = self.initialize_llm_client()

        # Publishers
        self.plan_pub = self.create_publisher(String, '/planning/plan', 10)
        self.action_pub = self.create_publisher(String, '/planning/action', 10)
        self.status_pub = self.create_publisher(String, '/planning/status', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        self.task_sub = self.create_subscription(
            String,
            '/high_level_task',
            self.task_callback,
            10
        )

        # Internal state
        self.current_plan = []
        self.executing_action = None
        self.plan_execution_status = 'idle'
        self.robot_capabilities = self.define_robot_capabilities()

        self.get_logger().info('LLM Cognitive Planner initialized')

    def initialize_llm_client(self):
        """
        Initialize LLM client (using OpenAI as example).
        """
        # Set your API key
        openai.api_key = "YOUR_API_KEY_HERE"  # In practice, use secure configuration

        return openai

    def define_robot_capabilities(self):
        """
        Define what the robot is capable of doing.
        """
        return {
            'navigation': {
                'abilities': ['move_forward', 'move_backward', 'turn_left', 'turn_right', 'go_to_location'],
                'constraints': ['max_speed: 0.5 m/s', 'max_rotation_speed: 1.0 rad/s']
            },
            'manipulation': {
                'abilities': ['grasp_object', 'release_object', 'pick_up', 'place_down'],
                'constraints': ['max_payload: 1.0 kg', 'reachable_area: 1.5m radius']
            },
            'perception': {
                'abilities': ['detect_objects', 'identify_colors', 'measure_distances'],
                'constraints': ['camera_fov: 60 degrees', 'depth_range: 0.1-3.0m']
            },
            'communication': {
                'abilities': ['speak', 'listen', 'display_text'],
                'constraints': ['language: English']
            }
        }

    def command_callback(self, msg):
        """
        Process high-level commands from voice or other sources.
        """
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Plan and execute
        asyncio.create_task(self.process_command_async(command))

    def task_callback(self, msg):
        """
        Process complex tasks that require cognitive planning.
        """
        task = msg.data
        self.get_logger().info(f'Received complex task: {task}')

        # Plan and execute complex task
        asyncio.create_task(self.plan_complex_task_async(task))

    async def process_command_async(self, command):
        """
        Process command asynchronously using LLM.
        """
        try:
            # Generate plan using LLM
            plan = await self.generate_plan_with_llm(command)

            # Execute plan
            await self.execute_plan(plan)

        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')

    async def generate_plan_with_llm(self, command):
        """
        Generate a plan using LLM based on the command.
        """
        # Define the system prompt with robot capabilities
        system_prompt = f"""
        You are a cognitive planning assistant for a robot. The robot has the following capabilities:

        {json.dumps(self.robot_capabilities, indent=2)}

        Your task is to break down high-level commands into executable actions.
        Respond with a JSON array of actions, where each action has:
        - 'action': the specific action to take
        - 'parameters': any required parameters
        - 'description': human-readable description of the action

        Actions should be atomic and executable by the robot.
        """

        # User command
        user_prompt = f"Plan the following command: {command}"

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm_client.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # or gpt-4 for more complex tasks
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1  # Low temperature for more deterministic output
                )
            )

            # Extract the plan from response
            plan_text = response.choices[0].message.content

            # Parse JSON response
            plan = self.parse_llm_response(plan_text)

            # Validate plan
            validated_plan = self.validate_plan(plan)

            self.get_logger().info(f'Generated plan: {validated_plan}')

            # Publish plan
            plan_msg = String()
            plan_msg.data = json.dumps(validated_plan)
            self.plan_pub.publish(plan_msg)

            return validated_plan

        except Exception as e:
            self.get_logger().error(f'Error generating plan with LLM: {e}')
            return []

    def parse_llm_response(self, response_text):
        """
        Parse LLM response to extract plan.
        """
        try:
            # Look for JSON in the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                plan = json.loads(json_str)
                return plan
            else:
                # If no JSON found, try to extract as plain text
                # This is a simplified fallback
                return [{"action": "unknown", "parameters": {}, "description": response_text}]
        except json.JSONDecodeError:
            self.get_logger().error('Could not parse LLM response as JSON')
            return []

    def validate_plan(self, plan):
        """
        Validate the plan against robot capabilities.
        """
        validated_plan = []

        for action in plan:
            action_type = action.get('action', '')

            # Check if action is supported by robot
            is_supported = False
            for capability_category in self.robot_capabilities.values():
                if action_type in capability_category['abilities']:
                    is_supported = True
                    break

            if is_supported:
                validated_plan.append(action)
            else:
                self.get_logger().warn(f'Action not supported by robot: {action_type}')
                # Could add fallback or error handling here

        return validated_plan

    async def execute_plan(self, plan):
        """
        Execute the generated plan step by step.
        """
        if not plan:
            self.get_logger().warn('No plan to execute')
            return

        self.plan_execution_status = 'executing'

        for i, action in enumerate(plan):
            self.get_logger().info(f'Executing action {i+1}/{len(plan)}: {action["description"]}')

            # Update status
            status_msg = String()
            status_msg.data = f'Executing: {action["description"]}'
            self.status_pub.publish(status_msg)

            # Execute action
            success = await self.execute_single_action(action)

            if not success:
                self.get_logger().error(f'Action failed: {action}')
                self.plan_execution_status = 'failed'
                break

        self.plan_execution_status = 'completed' if self.plan_execution_status == 'executing' else self.plan_execution_status

    async def execute_single_action(self, action):
        """
        Execute a single action from the plan.
        """
        action_type = action.get('action', '')
        parameters = action.get('parameters', {})

        try:
            if action_type == 'move_forward':
                return await self.execute_move_forward(parameters)
            elif action_type == 'move_backward':
                return await self.execute_move_backward(parameters)
            elif action_type == 'turn_left':
                return await self.execute_turn_left(parameters)
            elif action_type == 'turn_right':
                return await self.execute_turn_right(parameters)
            elif action_type == 'go_to_location':
                return await self.execute_go_to_location(parameters)
            elif action_type == 'grasp_object':
                return await self.execute_grasp_object(parameters)
            elif action_type == 'release_object':
                return await self.execute_release_object(parameters)
            elif action_type == 'detect_objects':
                return await self.execute_detect_objects(parameters)
            else:
                self.get_logger().warn(f'Unknown action type: {action_type}')
                return False

        except Exception as e:
            self.get_logger().error(f'Error executing action {action_type}: {e}')
            return False

    async def execute_move_forward(self, parameters):
        """
        Execute move forward action.
        """
        distance = parameters.get('distance', 1.0)  # default 1 meter
        speed = parameters.get('speed', 0.3)  # default 0.3 m/s

        # Publish action to ROS 2
        action_msg = String()
        action_msg.data = f"MOVE_FORWARD:{distance}:{speed}"
        self.action_pub.publish(action_msg)

        # Simulate execution time
        await asyncio.sleep(distance / speed)

        return True

    async def execute_move_backward(self, parameters):
        """
        Execute move backward action.
        """
        distance = parameters.get('distance', 1.0)
        speed = parameters.get('speed', 0.3)

        action_msg = String()
        action_msg.data = f"MOVE_BACKWARD:{distance}:{speed}"
        self.action_pub.publish(action_msg)

        await asyncio.sleep(distance / speed)

        return True

    async def execute_turn_left(self, parameters):
        """
        Execute turn left action.
        """
        angle = parameters.get('angle', 90.0)  # degrees
        speed = parameters.get('speed', 0.5)  # rad/s

        action_msg = String()
        action_msg.data = f"TURN_LEFT:{angle}:{speed}"
        self.action_pub.publish(action_msg)

        await asyncio.sleep(angle / 180.0 * 3.14159 / speed)

        return True

    async def execute_turn_right(self, parameters):
        """
        Execute turn right action.
        """
        angle = parameters.get('angle', 90.0)
        speed = parameters.get('speed', 0.5)

        action_msg = String()
        action_msg.data = f"TURN_RIGHT:{angle}:{speed}"
        self.action_pub.publish(action_msg)

        await asyncio.sleep(angle / 180.0 * 3.14159 / speed)

        return True

    async def execute_go_to_location(self, parameters):
        """
        Execute go to location action.
        """
        x = parameters.get('x', 0.0)
        y = parameters.get('y', 0.0)
        z = parameters.get('z', 0.0)

        # Create navigation goal
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = z

        # Publish navigation goal
        # self.nav_goal_pub.publish(goal)  # Would need navigation publisher

        action_msg = String()
        action_msg.data = f"GO_TO_LOCATION:{x},{y},{z}"
        self.action_pub.publish(action_msg)

        # Wait for navigation to complete (simplified)
        await asyncio.sleep(5.0)  # Simulated navigation time

        return True

    async def execute_grasp_object(self, parameters):
        """
        Execute grasp object action.
        """
        object_name = parameters.get('object', 'unknown')
        approach_height = parameters.get('approach_height', 0.1)

        action_msg = String()
        action_msg.data = f"GRASP_OBJECT:{object_name}:{approach_height}"
        self.action_pub.publish(action_msg)

        await asyncio.sleep(2.0)  # Simulated grasping time

        return True

    async def execute_release_object(self, parameters):
        """
        Execute release object action.
        """
        object_name = parameters.get('object', 'unknown')
        placement_height = parameters.get('placement_height', 0.0)

        action_msg = String()
        action_msg.data = f"RELEASE_OBJECT:{object_name}:{placement_height}"
        self.action_pub.publish(action_msg)

        await asyncio.sleep(2.0)  # Simulated releasing time

        return True

    async def execute_detect_objects(self, parameters):
        """
        Execute detect objects action.
        """
        detection_range = parameters.get('range', 2.0)
        object_types = parameters.get('object_types', ['all'])

        action_msg = String()
        action_msg.data = f"DETECT_OBJECTS:{detection_range}:{','.join(object_types)}"
        self.action_pub.publish(action_msg)

        await asyncio.sleep(1.0)  # Simulated detection time

        return True

    async def plan_complex_task_async(self, task):
        """
        Plan and execute a complex task requiring multiple steps.
        """
        self.get_logger().info(f'Planning complex task: {task}')

        # For complex tasks, we might need additional context
        context = {
            'robot_position': self.get_current_position(),
            'environment_map': self.get_environment_map(),
            'available_objects': self.get_available_objects(),
            'robot_state': self.get_robot_state()
        }

        # Generate complex plan with context
        plan = await self.generate_complex_plan_with_context(task, context)

        # Execute the complex plan
        await self.execute_plan(plan)

    def get_current_position(self):
        """
        Get current robot position (simplified).
        """
        # In real implementation, this would get from odometry
        return {'x': 0.0, 'y': 0.0, 'theta': 0.0}

    def get_environment_map(self):
        """
        Get environment map (simplified).
        """
        # In real implementation, this would get from SLAM or map server
        return {'known_locations': ['kitchen', 'living_room', 'bedroom']}

    def get_available_objects(self):
        """
        Get available objects in environment (simplified).
        """
        # In real implementation, this would get from perception system
        return ['red_ball', 'blue_cube', 'green_pyramid']

    def get_robot_state(self):
        """
        Get current robot state (simplified).
        """
        return {
            'battery_level': 85,
            'gripper_status': 'open',
            'navigation_status': 'ready'
        }

    async def generate_complex_plan_with_context(self, task, context):
        """
        Generate a complex plan using LLM with additional context.
        """
        system_prompt = f"""
        You are a cognitive planning assistant for a robot. The robot has these capabilities:

        {json.dumps(self.robot_capabilities, indent=2)}

        The current context is:
        - Robot Position: {context['robot_position']}
        - Environment Map: {context['environment_map']}
        - Available Objects: {context['available_objects']}
        - Robot State: {context['robot_state']}

        Plan the following complex task: {task}

        Break it down into executable steps, considering the current context.
        Respond with a JSON array of actions with 'action', 'parameters', and 'description'.
        """

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm_client.ChatCompletion.create(
                    model="gpt-4",  # Use more capable model for complex tasks
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task}
                    ],
                    temperature=0.1
                )
            )

            plan_text = response.choices[0].message.content
            plan = self.parse_llm_response(plan_text)
            validated_plan = self.validate_plan(plan)

            return validated_plan

        except Exception as e:
            self.get_logger().error(f'Error generating complex plan: {e}')
            return []
```

## Context-Aware Planning

### Maintaining World State

```python
class ContextAwarePlanner(LLMCognitivePlanner):
    """
    Planner that maintains context and world state for better planning.
    """

    def __init__(self):
        super().__init__()

        # Initialize context and world state
        self.world_state = {
            'locations': {},
            'objects': {},
            'robot_state': {},
            'task_history': [],
            'user_preferences': {}
        }

        # Timer for updating world state
        self.world_state_timer = self.create_timer(1.0, self.update_world_state)

    def update_world_state(self):
        """
        Update world state with latest information.
        """
        # Update robot state
        self.world_state['robot_state'] = self.get_updated_robot_state()

        # Update known locations from SLAM/map
        self.world_state['locations'] = self.get_updated_locations()

        # Update objects from perception system
        self.world_state['objects'] = self.get_updated_objects()

    def get_updated_robot_state(self):
        """
        Get latest robot state from sensors and systems.
        """
        # This would integrate with actual robot systems
        return {
            'position': self.get_current_position(),
            'battery': self.get_battery_level(),
            'gripper': self.get_gripper_status(),
            'navigation': self.get_navigation_status()
        }

    def get_updated_locations(self):
        """
        Get latest known locations from map/SLAM system.
        """
        # This would integrate with actual mapping system
        return self.get_environment_map()

    def get_updated_objects(self):
        """
        Get latest known objects from perception system.
        """
        # This would integrate with actual perception system
        return self.get_available_objects()

    async def generate_plan_with_context(self, command):
        """
        Generate plan using full world state context.
        """
        # Include world state in prompt
        context_str = json.dumps(self.world_state, indent=2)

        system_prompt = f"""
        You are a cognitive planning assistant for a robot. The robot has these capabilities:

        {json.dumps(self.robot_capabilities, indent=2)}

        Current world state:
        {context_str}

        Plan the following command: {command}

        Consider the current world state when generating the plan.
        Respond with a JSON array of actions with 'action', 'parameters', and 'description'.
        """

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm_client.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": command}
                    ],
                    temperature=0.1
                )
            )

            plan_text = response.choices[0].message.content
            plan = self.parse_llm_response(plan_text)
            validated_plan = self.validate_plan(plan)

            # Update task history
            self.world_state['task_history'].append({
                'command': command,
                'plan': validated_plan,
                'timestamp': time.time()
            })

            return validated_plan

        except Exception as e:
            self.get_logger().error(f'Error generating contextual plan: {e}')
            return []
```

## Plan Monitoring and Adjustment

### Adaptive Planning

```python
class AdaptivePlanner(ContextAwarePlanner):
    """
    Planner that can adapt and adjust plans based on execution feedback.
    """

    def __init__(self):
        super().__init__()

        # Subscribers for execution feedback
        self.execution_feedback_sub = self.create_subscription(
            String,
            '/execution_feedback',
            self.execution_feedback_callback,
            10
        )

        # Plan adjustment parameters
        self.adjustment_threshold = 0.7  # Confidence threshold for plan adjustments

    def execution_feedback_callback(self, msg):
        """
        Process execution feedback and adjust plan if needed.
        """
        try:
            feedback = json.loads(msg.data)
            action_result = feedback.get('action_result', 'success')
            action_id = feedback.get('action_id', '')
            confidence = feedback.get('confidence', 1.0)

            if action_result == 'failure' or confidence < self.adjustment_threshold:
                self.handle_execution_failure(action_id, feedback)

        except json.JSONDecodeError:
            self.get_logger().error('Could not parse execution feedback')

    def handle_execution_failure(self, action_id, feedback):
        """
        Handle action execution failure and adjust plan.
        """
        self.get_logger().warn(f'Action {action_id} failed: {feedback}')

        # Generate alternative plan using LLM
        async def adjust_plan():
            # Get current state and failed action info
            current_state = self.world_state.copy()
            failed_action = next((a for a in self.current_plan if a.get('id') == action_id), None)

            if failed_action:
                # Ask LLM for alternative approach
                alternative_plan = await self.generate_alternative_plan(
                    failed_action, current_state, feedback
                )

                if alternative_plan:
                    # Replace failed action with alternative
                    self.replace_failed_action(action_id, alternative_plan)

        # Execute adjustment asynchronously
        asyncio.create_task(adjust_plan())

    async def generate_alternative_plan(self, failed_action, current_state, feedback):
        """
        Generate alternative plan when original action fails.
        """
        system_prompt = f"""
        You are a cognitive planning assistant for a robot. The robot has these capabilities:

        {json.dumps(self.robot_capabilities, indent=2)}

        Current world state:
        {json.dumps(current_state, indent=2)}

        The following action failed:
        {json.dumps(failed_action, indent=2)}

        Failure feedback:
        {json.dumps(feedback, indent=2)}

        Generate an alternative plan to achieve the same goal, considering why the original action failed.
        Respond with a JSON array of alternative actions.
        """

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm_client.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Generate alternative actions to achieve the goal"}
                    ],
                    temperature=0.2
                )
            )

            plan_text = response.choices[0].message.content
            alternative_plan = self.parse_llm_response(plan_text)
            validated_plan = self.validate_plan(alternative_plan)

            return validated_plan

        except Exception as e:
            self.get_logger().error(f'Error generating alternative plan: {e}')
            return []

    def replace_failed_action(self, action_id, alternative_plan):
        """
        Replace a failed action with an alternative plan.
        """
        # Find and replace the failed action in the current plan
        for i, action in enumerate(self.current_plan):
            if action.get('id') == action_id:
                # Replace with alternative actions
                self.current_plan = (
                    self.current_plan[:i] +
                    alternative_plan +
                    self.current_plan[i+1:]
                )

                self.get_logger().info(f'Replaced failed action {action_id} with alternative plan')
                break
```

## Safety and Validation

### Plan Safety Checker

```python
class SafeLLMPlanner(AdaptivePlanner):
    """
    Planner with built-in safety checks and validation.
    """

    def __init__(self):
        super().__init__()

        # Safety constraints
        self.safety_constraints = {
            'collision_avoidance': True,
            'workspace_boundaries': True,
            'payload_limits': True,
            'energy_efficiency': True
        }

    def validate_plan(self, plan):
        """
        Validate plan against safety constraints.
        """
        validated_plan = []

        for action in plan:
            if self.is_action_safe(action):
                validated_plan.append(action)
            else:
                self.get_logger().warn(f'Action deemed unsafe: {action}')
                # Could implement safe alternatives here

        return validated_plan

    def is_action_safe(self, action):
        """
        Check if an action is safe to execute.
        """
        action_type = action.get('action', '')
        parameters = action.get('parameters', {})

        # Check various safety constraints
        if not self.check_collision_risk(action_type, parameters):
            return False

        if not self.check_workspace_boundary(action_type, parameters):
            return False

        if not self.check_payload_limit(action_type, parameters):
            return False

        return True

    def check_collision_risk(self, action_type, parameters):
        """
        Check if action poses collision risk.
        """
        # This would integrate with collision detection system
        if action_type in ['move_forward', 'move_backward', 'go_to_location']:
            # Check path for obstacles
            target_location = self.calculate_target_location(action_type, parameters)
            if self.has_obstacles_in_path(target_location):
                return False

        return True

    def check_workspace_boundary(self, action_type, parameters):
        """
        Check if action respects workspace boundaries.
        """
        if action_type in ['go_to_location']:
            target_x = parameters.get('x', 0.0)
            target_y = parameters.get('y', 0.0)

            # Check if target is within workspace boundaries
            if (abs(target_x) > 10.0 or abs(target_y) > 10.0):  # Example boundaries
                return False

        return True

    def check_payload_limit(self, action_type, parameters):
        """
        Check if action respects payload limits.
        """
        if action_type in ['grasp_object', 'pick_up']:
            object_weight = parameters.get('weight', 0.0)
            max_payload = 1.0  # kg

            if object_weight > max_payload:
                return False

        return True

    def calculate_target_location(self, action_type, parameters):
        """
        Calculate target location for navigation actions.
        """
        # Simplified calculation
        if action_type == 'go_to_location':
            return (parameters.get('x', 0.0), parameters.get('y', 0.0))
        elif action_type == 'move_forward':
            # Calculate based on current position and distance
            current_pos = self.get_current_position()
            distance = parameters.get('distance', 1.0)
            # Simplified forward movement calculation
            return (current_pos['x'] + distance, current_pos['y'])
        else:
            return (0.0, 0.0)

    def has_obstacles_in_path(self, target_location):
        """
        Check if there are obstacles in the path to target location.
        """
        # This would integrate with perception/collision detection
        # For now, return False (no obstacles detected)
        return False
```

## Integration with Other Systems

### Multi-Agent Coordination

```python
class MultiAgentLLMPlanner(SafeLLMPlanner):
    """
    Planner that can coordinate with other agents/robots.
    """

    def __init__(self):
        super().__init__()

        # Publisher for coordination
        self.coordination_pub = self.create_publisher(
            String,
            '/coordination_requests',
            10
        )

        # Subscriber for coordination feedback
        self.coordination_sub = self.create_subscription(
            String,
            '/coordination_responses',
            self.coordination_callback,
            10
        )

        # Track other agents
        self.other_agents = {}

    def coordination_callback(self, msg):
        """
        Handle coordination messages from other agents.
        """
        try:
            coordination_data = json.loads(msg.data)
            agent_id = coordination_data.get('agent_id')
            request_type = coordination_data.get('request_type')
            content = coordination_data.get('content')

            if request_type == 'resource_request':
                self.handle_resource_request(agent_id, content)
            elif request_type == 'task_coordination':
                self.handle_task_coordination(agent_id, content)

        except json.JSONDecodeError:
            self.get_logger().error('Could not parse coordination message')

    def handle_resource_request(self, agent_id, content):
        """
        Handle resource request from another agent.
        """
        resource = content.get('resource')
        priority = content.get('priority', 'normal')

        # Decide whether to grant resource request
        if self.can_grant_resource(resource, priority):
            response = {
                'agent_id': self.get_namespace(),  # This robot's ID
                'request_id': content.get('request_id'),
                'response': 'granted',
                'resource': resource
            }
        else:
            response = {
                'agent_id': self.get_namespace(),
                'request_id': content.get('request_id'),
                'response': 'denied',
                'reason': 'resource_not_available'
            }

        # Publish response
        response_msg = String()
        response_msg.data = json.dumps(response)
        self.coordination_pub.publish(response_msg)

    def can_grant_resource(self, resource, priority):
        """
        Determine if we can grant a resource request.
        """
        # Check if resource is currently needed by this agent
        if resource == 'navigation_space' and self.plan_execution_status == 'executing':
            return priority == 'emergency'  # Only grant in emergencies

        return True  # Simplified - in practice, check resource availability

    async def generate_coordinated_plan(self, task, other_agents_info):
        """
        Generate plan considering other agents in the environment.
        """
        system_prompt = f"""
        You are a cognitive planning assistant for a robot coordinating with other agents.
        The robot has these capabilities:

        {json.dumps(self.robot_capabilities, indent=2)}

        Other agents in the environment:
        {json.dumps(other_agents_info, indent=2)}

        Current world state:
        {json.dumps(self.world_state, indent=2)}

        Plan the following task while considering coordination with other agents: {task}

        Be aware of potential resource conflicts and coordinate appropriately.
        Respond with a JSON array of actions.
        """

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm_client.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task}
                    ],
                    temperature=0.1
                )
            )

            plan_text = response.choices[0].message.content
            plan = self.parse_llm_response(plan_text)
            validated_plan = self.validate_plan(plan)

            return validated_plan

        except Exception as e:
            self.get_logger().error(f'Error generating coordinated plan: {e}')
            return []
```

## Best Practices for LLM Integration

### Performance and Reliability

1. **Caching**: Cache common plans and responses to reduce LLM calls
2. **Fallback Mechanisms**: Implement traditional planners as fallback
3. **Timeout Handling**: Set reasonable timeouts for LLM responses
4. **Error Recovery**: Plan for LLM failures gracefully
5. **Context Management**: Carefully manage context to avoid confusion
6. **Security**: Secure API keys and sensitive data
7. **Privacy**: Consider privacy implications of sending data to LLMs
8. **Cost Management**: Monitor and control API usage costs

### Testing and Validation

1. **Unit Tests**: Test individual planning components
2. **Integration Tests**: Test full planning-execution loop
3. **Simulation Testing**: Test in simulated environments first
4. **Safety Testing**: Verify safety constraints are enforced
5. **Edge Case Testing**: Test unusual or unexpected commands

LLM-based cognitive planning opens up new possibilities for flexible, adaptive robotic systems that can understand and execute complex, natural language commands while adapting to dynamic environments and unexpected situations.
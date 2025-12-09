# Contract: ROS 2 Humanoid Joint Control Service

## Purpose
To control specific joints of a humanoid robot via a ROS 2 service.

## ROS 2 Service
- `/robot/set_joint_position`

## Request Message Type
- `robot_control_msgs/SetJointPosition.srv` (conceptual)

## Inputs (Request)
- `joint_name`: (string) Name of the joint to control (e.g., "right_shoulder_pitch_joint").
- `position`: (float) Target position for the joint in radians.
- `velocity`: (float, optional) Desired velocity for the joint movement.

## Response Message Type
- `robot_control_msgs/SetJointPosition.srv` (conceptual)

## Outputs (Response)
- `success`: (boolean) True if the joint command was accepted, false otherwise.
- `message`: (string) A descriptive message, especially if `success` is false (e.g., "Joint not found", "Position out of limits").

## Error Taxonomy
- `JOINT_NOT_FOUND`: The specified `joint_name` does not exist.
- `INVALID_POSITION`: The `position` value is outside the joint's physical limits.
- `CONTROLLER_ERROR`: An internal error occurred in the robot's joint controller.

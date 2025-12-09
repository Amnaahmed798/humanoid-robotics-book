# Contract: ROS 2 Robotic Arm Movement

## Purpose
To demonstrate basic control of a robotic arm using ROS 2 topics.

## ROS 2 Topic
- `/robot/arm_commands`

## Message Type
- `geometry_msgs/Twist` (conceptual for joint velocity/position commands)

## Inputs
- `linear.x`: (float) Desired linear velocity for the end-effector along X-axis.
- `angular.z`: (float) Desired angular velocity for a specific joint (e.g., base rotation).

## Outputs
- None (commands are sent, robot executes asynchronously).

## Error Taxonomy
- No explicit error responses for topic publishing; robot controller handles errors internally.

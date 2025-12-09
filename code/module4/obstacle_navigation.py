#!/usr/bin/env python3

"""
Advanced Obstacle Navigation for Humanoid Robot

This module implements advanced obstacle navigation algorithms for humanoid robots,
including path planning, dynamic obstacle avoidance, and multi-modal perception
integration for robust navigation in complex environments.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import LaserScan, PointCloud2, Image, CameraInfo
from geometry_msgs.msg import Twist, Point, Pose, PoseStamped, Vector3
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
import numpy as np
import cv2
from cv_bridge import CvBridge
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import threading
import time
from enum import Enum


class NavigationState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    AVOIDING = "avoiding"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class Obstacle:
    """
    Data structure for detected obstacles.
    """
    id: str
    position: Point
    size: Vector3  # Width, depth, height
    velocity: Vector3  # Velocity vector
    confidence: float
    is_dynamic: bool = False
    last_seen: float = 0.0


@dataclass
class NavigationGoal:
    """
    Data structure for navigation goals.
    """
    position: Point
    orientation: float  # Yaw angle
    frame_id: str = "map"
    tolerance: float = 0.2  # Tolerance for reaching goal


class ObstacleNavigationNode(Node):
    """
    Advanced obstacle navigation node for humanoid robots.
    """

    def __init__(self):
        super().__init__('obstacle_navigation_node')

        # CV Bridge for image processing
        self.cv_bridge = CvBridge()

        # TF2 for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.global_plan_pub = self.create_publisher(Path, '/global_plan', 10)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)
        self.obstacle_pub = self.create_publisher(MarkerArray, '/obstacles', 10)
        self.status_pub = self.create_publisher(String, '/navigation/status', 10)
        self.velocity_limit_pub = self.create_publisher(Float32, '/velocity_limit', 10)

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/pointcloud',
            self.pointcloud_callback,
            10
        )

        self.camera_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.camera_callback,
            10
        )

        # Internal state
        self.current_pose = Pose()
        self.current_twist = Twist()
        self.current_goal: Optional[NavigationGoal] = None
        self.obstacles: List[Obstacle] = []
        self.navigation_state = NavigationState.IDLE
        self.path = []  # Planned path as list of Points
        self.current_path_index = 0

        # Navigation parameters
        self.linear_speed = 0.3  # m/s
        self.angular_speed = 0.5  # rad/s
        self.min_obstacle_distance = 0.5  # meters
        self.arrival_threshold = 0.3  # meters
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 1.0  # rad/s
        self.safety_margin = 0.3  # meters

        # Grid map parameters
        self.grid_resolution = 0.1  # meters per cell
        self.grid_size = 100  # cells per side (10m x 10m grid)
        self.grid_center = (50, 50)  # Center of grid in grid coordinates
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Dynamic obstacle tracking
        self.dynamic_obstacles = {}
        self.obstacle_history = {}
        self.obstacle_prediction_horizon = 2.0  # seconds

        # Path planning parameters
        self.planning_frequency = 1.0  # Hz
        self.replan_distance = 0.5  # meters from current path
        self.clearance_threshold = 0.2  # meters for obstacle clearance

        # Timers
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10Hz control
        self.planning_timer = self.create_timer(1.0/self.planning_frequency, self.plan_path)
        self.visualization_timer = self.create_timer(0.5, self.publish_visualization)

        self.get_logger().info('Obstacle Navigation Node initialized')

    def laser_callback(self, msg):
        """
        Process LiDAR data to detect obstacles.
        """
        try:
            # Convert laser scan to obstacle positions
            obstacle_positions = self.laser_to_obstacles(msg)

            # Update obstacle list
            self.update_obstacles(obstacle_positions, is_dynamic=False)

        except Exception as e:
            self.get_logger().error(f'Error processing laser scan: {e}')

    def odom_callback(self, msg):
        """
        Update robot pose and twist from odometry.
        """
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def goal_callback(self, msg):
        """
        Set new navigation goal.
        """
        goal = NavigationGoal(
            position=msg.pose.position,
            orientation=self.quaternion_to_yaw(msg.pose.orientation),
            frame_id=msg.header.frame_id
        )

        self.current_goal = goal
        self.navigation_state = NavigationState.PLANNING

        self.get_logger().info(f'New goal received: ({goal.position.x:.2f}, {goal.position.y:.2f})')

    def pointcloud_callback(self, msg):
        """
        Process point cloud data for detailed obstacle detection.
        """
        # This would convert PointCloud2 to 3D obstacles
        # For now, we'll use a placeholder
        pass

    def camera_callback(self, msg):
        """
        Process camera image for visual obstacle detection.
        """
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Detect obstacles in image (simplified - in practice, use object detection)
            visual_obstacles = self.detect_visual_obstacles(cv_image)

            # Transform visual obstacles to world coordinates
            world_obstacles = self.transform_visual_obstacles(visual_obstacles)

            # Update obstacle list
            self.update_obstacles(world_obstacles, is_dynamic=True)

        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {e}')

    def laser_to_obstacles(self, laser_msg: LaserScan) -> List[Point]:
        """
        Convert laser scan to obstacle positions.
        """
        obstacles = []
        angle = laser_msg.angle_min

        for i, range_val in enumerate(laser_msg.ranges):
            if laser_msg.range_min <= range_val <= laser_msg.range_max:
                # Convert polar to Cartesian
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)

                # Transform to map frame
                obstacle_point = Point(x=x, y=y, z=0.0)
                obstacles.append(obstacle_point)

            angle += laser_msg.angle_increment

        return obstacles

    def detect_visual_obstacles(self, image) -> List[Dict]:
        """
        Detect obstacles in camera image (simplified implementation).
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold to detect obstacles
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        visual_obstacles = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Minimum area threshold
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                obstacle_info = {
                    'pixel_x': x + w/2,
                    'pixel_y': y + h/2,
                    'width': w,
                    'height': h
                }
                visual_obstacles.append(obstacle_info)

        return visual_obstacles

    def transform_visual_obstacles(self, visual_obstacles) -> List[Point]:
        """
        Transform visual obstacles from pixel coordinates to world coordinates.
        """
        world_obstacles = []

        for obs in visual_obstacles:
            # This would use camera calibration and robot pose to convert
            # For now, return a placeholder
            world_point = Point(x=obs['pixel_x']/100.0, y=obs['pixel_y']/100.0, z=0.0)
            world_obstacles.append(world_point)

        return world_obstacles

    def update_obstacles(self, new_obstacles: List[Point], is_dynamic: bool):
        """
        Update obstacle list with new detections.
        """
        current_time = time.time()

        for point in new_obstacles:
            # Check if this obstacle is already tracked
            closest_existing = self.find_closest_obstacle(point)

            if closest_existing and self.distance_3d(closest_existing.position, point) < 0.3:
                # Update existing obstacle
                closest_existing.last_seen = current_time
                closest_existing.is_dynamic = is_dynamic or closest_existing.is_dynamic

                # Update velocity if it's dynamic
                if is_dynamic and closest_existing.id in self.obstacle_history:
                    prev_pos = self.obstacle_history[closest_existing.id][-1]
                    dt = current_time - closest_existing.last_seen
                    if dt > 0:
                        closest_existing.velocity.x = (point.x - prev_pos.x) / dt
                        closest_existing.velocity.y = (point.y - prev_pos.y) / dt
            else:
                # Create new obstacle
                obstacle_id = f'obs_{len(self.obstacles)}'
                new_obstacle = Obstacle(
                    id=obstacle_id,
                    position=point,
                    size=Vector3(x=0.2, y=0.2, z=0.5),  # Default size
                    velocity=Vector3(x=0.0, y=0.0, z=0.0),
                    confidence=0.8,
                    is_dynamic=is_dynamic,
                    last_seen=current_time
                )
                self.obstacles.append(new_obstacle)

            # Update obstacle history
            if is_dynamic:
                if new_obstacle.id not in self.obstacle_history:
                    self.obstacle_history[new_obstacle.id] = []
                self.obstacle_history[new_obstacle.id].append(point)

        # Remove old obstacles (not seen for more than 3 seconds)
        self.obstacles = [
            obs for obs in self.obstacles
            if current_time - obs.last_seen < 3.0
        ]

    def find_closest_obstacle(self, point: Point) -> Optional[Obstacle]:
        """
        Find the closest existing obstacle to the given point.
        """
        if not self.obstacles:
            return None

        closest = None
        min_dist = float('inf')

        for obstacle in self.obstacles:
            dist = self.distance_3d(obstacle.position, point)
            if dist < min_dist:
                min_dist = dist
                closest = obstacle

        return closest if min_dist < 0.5 else None  # Only match if close enough

    def distance_3d(self, p1: Point, p2: Point) -> float:
        """
        Calculate 3D Euclidean distance between two points.
        """
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def quaternion_to_yaw(self, quaternion) -> float:
        """
        Convert quaternion to yaw angle.
        """
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def plan_path(self):
        """
        Plan path using A* algorithm with obstacle avoidance.
        """
        if self.navigation_state != NavigationState.PLANNING or self.current_goal is None:
            return

        start = self.current_pose.position
        goal = self.current_goal.position

        # Create occupancy grid
        self.update_occupancy_grid()

        # Plan path using A*
        path = self.a_star_planning(start, goal)

        if path:
            self.path = path
            self.current_path_index = 0
            self.navigation_state = NavigationState.EXECUTING

            # Publish global plan
            self.publish_global_plan()

            self.get_logger().info(f'Path planned with {len(path)} waypoints')
        else:
            self.get_logger().error('Could not find a path to the goal')
            self.navigation_state = NavigationState.ERROR

    def update_occupancy_grid(self):
        """
        Update occupancy grid based on current obstacles.
        """
        # Clear grid
        self.grid.fill(0)

        # Add obstacles to grid
        for obstacle in self.obstacles:
            grid_x = int((obstacle.position.x - self.current_pose.position.x) / self.grid_resolution) + self.grid_center[0]
            grid_y = int((obstacle.position.y - self.current_pose.position.y) / self.grid_resolution) + self.grid_center[1]

            # Add safety margin around obstacles
            margin_cells = int(self.safety_margin / self.grid_resolution)

            for dx in range(-margin_cells, margin_cells + 1):
                for dy in range(-margin_cells, margin_cells + 1):
                    x_idx = grid_x + dx
                    y_idx = grid_y + dy

                    if 0 <= x_idx < self.grid_size and 0 <= y_idx < self.grid_size:
                        self.grid[y_idx, x_idx] = 100  # Occupied

    def a_star_planning(self, start: Point, goal: Point) -> Optional[List[Point]]:
        """
        A* path planning algorithm.
        """
        # Convert world coordinates to grid coordinates
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)

        if not self.is_valid_cell(start_grid[0], start_grid[1]) or not self.is_valid_cell(goal_grid[0], goal_grid[1]):
            return None

        # A* algorithm
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    world_point = self.grid_to_world(current[0], current[1])
                    path.append(world_point)
                    current = came_from[current]
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + self.distance_2d(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)

                    # Add to open set if not already there
                    if not any(neighbor == item[1] for item in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def world_to_grid(self, point: Point) -> Tuple[int, int]:
        """
        Convert world coordinates to grid coordinates.
        """
        grid_x = int((point.x - self.current_pose.position.x) / self.grid_resolution) + self.grid_center[0]
        grid_y = int((point.y - self.current_pose.position.y) / self.grid_resolution) + self.grid_center[1]

        return (max(0, min(self.grid_size - 1, grid_x)),
                max(0, min(self.grid_size - 1, grid_y)))

    def grid_to_world(self, grid_x: int, grid_y: int) -> Point:
        """
        Convert grid coordinates to world coordinates.
        """
        world_x = (grid_x - self.grid_center[0]) * self.grid_resolution + self.current_pose.position.x
        world_y = (grid_y - self.grid_center[1]) * self.grid_resolution + self.current_pose.position.y

        return Point(x=world_x, y=world_y, z=0.0)

    def is_valid_cell(self, x: int, y: int) -> bool:
        """
        Check if grid cell is valid (not occupied).
        """
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False

        return self.grid[y, x] < 50  # Threshold for occupancy

    def get_neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighboring cells.
        """
        neighbors = []
        x, y = cell

        # 8-connected neighborhood
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip current cell

                nx, ny = x + dx, y + dy
                if self.is_valid_cell(nx, ny):
                    neighbors.append((nx, ny))

        return neighbors

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Heuristic function for A* (Manhattan distance).
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def distance_2d(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Calculate 2D distance between grid cells.
        """
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def control_loop(self):
        """
        Main control loop for navigation.
        """
        if self.navigation_state == NavigationState.IDLE:
            # Publish zero velocity
            self.stop_robot()
            return

        if self.navigation_state == NavigationState.ERROR:
            # Emergency stop
            self.stop_robot()
            return

        if self.navigation_state == NavigationState.EXECUTING:
            # Execute planned path
            if self.follow_path():
                # Goal reached
                self.navigation_state = NavigationState.IDLE
                self.current_goal = None
                self.path = []
                self.get_logger().info('Goal reached successfully!')
                self.publish_status('Goal reached')
            else:
                # Check for obstacles and adjust path if needed
                self.check_and_avoid_obstacles()

        elif self.navigation_state == NavigationState.AVOIDING:
            # Execute obstacle avoidance behavior
            self.execute_avoidance()

    def follow_path(self) -> bool:
        """
        Follow the planned path, return True if goal reached.
        """
        if not self.path or self.current_path_index >= len(self.path):
            return True  # Goal reached

        # Get current target waypoint
        target = self.path[self.current_path_index]

        # Calculate distance to target
        dx = target.x - self.current_pose.position.x
        dy = target.y - self.current_pose.position.y
        distance_to_waypoint = math.sqrt(dx**2 + dy**2)

        # Check if we've reached the current waypoint
        if distance_to_waypoint < self.arrival_threshold:
            self.current_path_index += 1
            if self.current_path_index >= len(self.path):
                # Check if we're close to final goal
                goal_dx = self.current_goal.position.x - self.current_pose.position.x
                goal_dy = self.current_goal.position.y - self.current_pose.position.y
                goal_distance = math.sqrt(goal_dx**2 + goal_dy**2)
                return goal_distance < self.arrival_threshold

        # Calculate desired direction
        target_angle = math.atan2(dy, dx)

        # Get current robot angle
        current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)

        # Calculate angle difference
        angle_diff = self.normalize_angle(target_angle - current_yaw)

        # Create velocity command
        cmd_vel = Twist()

        # Move forward if aligned with path
        if abs(angle_diff) < 0.3:  # 0.3 rad = ~17 degrees
            cmd_vel.linear.x = min(self.linear_speed, distance_to_waypoint)
            cmd_vel.angular.z = 0.0
        else:
            # Rotate to face target direction
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = self.angular_speed * self.sign(angle_diff)

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

        return False  # Goal not reached yet

    def check_and_avoid_obstacles(self):
        """
        Check for obstacles in the path and initiate avoidance if needed.
        """
        # Check if there are obstacles blocking the path
        if self.is_path_blocked():
            self.navigation_state = NavigationState.AVOIDING
            self.get_logger().info('Obstacle detected in path, initiating avoidance')

    def is_path_blocked(self) -> bool:
        """
        Check if current path is blocked by obstacles.
        """
        if not self.path or self.current_path_index >= len(self.path):
            return False

        # Check next few waypoints for obstacles
        look_ahead = min(5, len(self.path) - self.current_path_index)

        for i in range(self.current_path_index, self.current_path_index + look_ahead):
            if i < len(self.path):
                waypoint = self.path[i]

                # Check for obstacles near this waypoint
                for obstacle in self.obstacles:
                    dist = self.distance_3d(waypoint, obstacle.position)
                    if dist < self.min_obstacle_distance:
                        return True

        # Check along the path segments
        for i in range(self.current_path_index, min(len(self.path) - 1, self.current_path_index + look_ahead)):
            seg_start = self.path[i]
            seg_end = self.path[i + 1]

            for obstacle in self.obstacles:
                if self.is_point_near_segment(obstacle.position, seg_start, seg_end, self.min_obstacle_distance):
                    return True

        return False

    def is_point_near_segment(self, point: Point, seg_start: Point, seg_end: Point, threshold: float) -> bool:
        """
        Check if a point is near a line segment.
        """
        # Vector from segment start to end
        seg_vec = Point(x=seg_end.x - seg_start.x, y=seg_end.y - seg_start.y, z=0)
        seg_len_sq = seg_vec.x**2 + seg_vec.y**2

        if seg_len_sq == 0:
            # Segment is actually a point
            dist_sq = (point.x - seg_start.x)**2 + (point.y - seg_start.y)**2
            return dist_sq <= threshold**2

        # Project point onto segment
        t = max(0, min(1, ((point.x - seg_start.x) * seg_vec.x + (point.y - seg_start.y) * seg_vec.y) / seg_len_sq))

        # Find closest point on segment
        closest = Point(
            x=seg_start.x + t * seg_vec.x,
            y=seg_start.y + t * seg_vec.y,
            z=0
        )

        # Calculate distance to closest point
        dist_sq = (point.x - closest.x)**2 + (point.y - closest.y)**2
        return dist_sq <= threshold**2

    def execute_avoidance(self):
        """
        Execute obstacle avoidance behavior.
        """
        # Simple reactive avoidance
        cmd_vel = Twist()

        # Check for obstacles in front
        front_clear = True
        for obstacle in self.obstacles:
            # Check if obstacle is in front of robot
            dx = obstacle.position.x - self.current_pose.position.x
            dy = obstacle.position.y - self.current_pose.position.y
            distance = math.sqrt(dx**2 + dy**2)

            if distance < self.min_obstacle_distance:
                # Calculate angle to obstacle
                angle_to_obstacle = math.atan2(dy, dx)
                current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)
                angle_diff = self.normalize_angle(angle_to_obstacle - current_yaw)

                # If obstacle is in front (+/- 90 degrees)
                if abs(angle_diff) < math.pi / 2:
                    front_clear = False
                    # Turn away from obstacle
                    cmd_vel.angular.z = self.angular_speed * self.sign(-angle_diff)
                    cmd_vel.linear.x = 0.0  # Stop moving forward
                    break

        if front_clear:
            # Path seems clear, resume following
            cmd_vel.linear.x = self.linear_speed * 0.5  # Slow down while resuming
            cmd_vel.angular.z = 0.0
            self.navigation_state = NavigationState.EXECUTING

        self.cmd_vel_pub.publish(cmd_vel)

    def normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi] range.
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def sign(self, value):
        """
        Return sign of value.
        """
        return 1.0 if value >= 0.0 else -1.0

    def stop_robot(self):
        """
        Stop the robot by publishing zero velocities.
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)

    def publish_global_plan(self):
        """
        Publish the global path for visualization.
        """
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for point in self.path:
            pose = PoseStamped()
            pose.pose.position = point
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.global_plan_pub.publish(path_msg)

    def publish_visualization(self):
        """
        Publish obstacle visualization markers.
        """
        marker_array = MarkerArray()

        for i, obstacle in enumerate(self.obstacles):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'obstacles'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position = obstacle.position
            marker.pose.orientation.w = 1.0

            marker.scale = obstacle.size
            marker.color.r = 1.0 if obstacle.is_dynamic else 0.0
            marker.color.g = 0.0
            marker.color.b = 0.0 if obstacle.is_dynamic else 1.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        self.obstacle_pub.publish(marker_array)

    def publish_status(self, status_text: str):
        """
        Publish navigation status.
        """
        status_msg = String()
        status_msg.data = status_text
        self.status_pub.publish(status_msg)


def main(args=None):
    """
    Main function to run the obstacle navigation system.
    """
    rclpy.init(args=args)

    node = ObstacleNavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down obstacle navigation node...')
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
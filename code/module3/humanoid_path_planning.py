#!/usr/bin/env python3

"""
Humanoid Path Planning Pipeline using NVIDIA Isaac

This example demonstrates a path planning pipeline for humanoid robots using
NVIDIA Isaac SDK components. The pipeline includes environment representation,
path planning algorithms, and trajectory generation for humanoid locomotion.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
import tf_transformations
from scipy.spatial import distance
import matplotlib.pyplot as plt
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient


class HumanoidPathPlanner(Node):
    """
    Path planning node for humanoid robots with support for bipedal locomotion.
    """

    def __init__(self):
        super().__init__('humanoid_path_planner')

        # Publishers
        self.path_pub = self.create_publisher(Path, '/humanoid_path', 10)
        self.trajectory_pub = self.create_publisher(Path, '/humanoid_trajectory', 10)
        self.visualization_pub = self.create_publisher(MarkerArray, '/path_visualization', 10)

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10
        )

        # Action client for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Internal state
        self.map_data = None
        self.map_resolution = 0.05  # meters per cell
        self.map_width = 0
        self.map_height = 0
        self.map_origin = [0.0, 0.0, 0.0]  # x, y, theta
        self.start_pose = None
        self.goal_pose = None
        self.current_pose = None

        # Path planning parameters
        self.robot_radius = 0.3  # meters (for collision checking)
        self.step_size = 0.1  # meters (discretization for path planning)
        self.max_step_angle = 0.5  # radians (max step angle for humanoid)
        self.min_step_distance = 0.2  # minimum distance between footsteps

        # Initialize
        self.get_logger().info('Humanoid Path Planner initialized')

    def map_callback(self, msg):
        """
        Process occupancy grid map message.
        """
        self.map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin = [
            msg.info.origin.position.x,
            msg.info.origin.position.y,
            tf_transformations.euler_from_quaternion([
                msg.info.origin.orientation.x,
                msg.info.origin.orientation.y,
                msg.info.origin.orientation.z,
                msg.info.origin.orientation.w
            ])[2]
        ]

        self.get_logger().info(
            f'Map received: {self.map_width}x{self.map_height}, '
            f'resolution: {self.map_resolution}m'
        )

    def goal_callback(self, msg):
        """
        Process goal pose message and plan path.
        """
        self.goal_pose = msg.pose
        self.get_logger().info(
            f'New goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})'
        )

        # Plan path if we have map data
        if self.map_data is not None:
            self.plan_path()

    def plan_path(self):
        """
        Plan a path from current position to goal using RRT* algorithm
        adapted for humanoid locomotion constraints.
        """
        if self.current_pose is None:
            self.get_logger().warn('Current pose unknown, cannot plan path')
            return

        # Convert poses to grid coordinates
        start_grid = self.world_to_grid(
            self.current_pose.position.x,
            self.current_pose.position.y
        )
        goal_grid = self.world_to_grid(
            self.goal_pose.position.x,
            self.goal_pose.position.y
        )

        # Check if start and goal are valid
        if not self.is_valid_cell(start_grid[0], start_grid[1]):
            self.get_logger().error('Start position is in obstacle')
            return

        if not self.is_valid_cell(goal_grid[0], goal_grid[1]):
            self.get_logger().error('Goal position is in obstacle')
            return

        # Plan path using RRT* adapted for humanoid
        path = self.rrt_star_humonoid(start_grid, goal_grid)

        if path:
            # Convert grid path back to world coordinates
            world_path = []
            for grid_cell in path:
                world_x, world_y = self.grid_to_world(grid_cell[0], grid_cell[1])
                world_path.append((world_x, world_y))

            # Smooth the path considering humanoid constraints
            smoothed_path = self.smooth_humonoid_path(world_path)

            # Generate trajectory with humanoid-specific constraints
            trajectory = self.generate_humonoid_trajectory(smoothed_path)

            # Publish path and trajectory
            self.publish_path(smoothed_path)
            self.publish_trajectory(trajectory)
            self.visualize_path(smoothed_path, trajectory)
        else:
            self.get_logger().error('No path found to goal')

    def rrt_star_humonoid(self, start, goal):
        """
        RRT* algorithm adapted for humanoid locomotion constraints.
        """
        # Define RRT* parameters
        max_iterations = 10000
        goal_bias = 0.1
        step_size = int(1.0 / self.map_resolution)  # Convert meters to grid cells

        # Initialize tree
        tree = {start: None}
        costs = {start: 0}

        for iteration in range(max_iterations):
            # Sample random point
            if np.random.random() < goal_bias:
                rand_point = goal
            else:
                rand_point = self.sample_random_point()

            # Find nearest node in tree
            nearest = self.nearest_node(tree, rand_point)

            # Steer towards random point
            new_point = self.steer(nearest, rand_point, step_size)

            # Check if new point is valid
            if self.is_valid_connection(nearest, new_point):
                # Add to tree
                tree[new_point] = nearest
                costs[new_point] = costs[nearest] + self.distance(nearest, new_point)

                # Rewire tree (RRT* improvement)
                self.rewire_tree(tree, costs, new_point, step_size)

                # Check if goal is reached
                if self.distance(new_point, goal) < step_size * 2:
                    # Reconstruct path
                    path = self.reconstruct_path(tree, start, new_point)
                    return path

        return None  # No path found

    def sample_random_point(self):
        """
        Sample a random point in the map.
        """
        x = np.random.randint(0, self.map_width)
        y = np.random.randint(0, self.map_height)
        return (x, y)

    def nearest_node(self, tree, point):
        """
        Find the nearest node in the tree to the given point.
        """
        min_dist = float('inf')
        nearest = None

        for node in tree:
            dist = self.distance(node, point)
            if dist < min_dist:
                min_dist = dist
                nearest = node

        return nearest

    def steer(self, from_point, to_point, step_size):
        """
        Steer from one point towards another with given step size.
        """
        dx = to_point[0] - from_point[0]
        dy = to_point[1] - from_point[1]
        dist = np.sqrt(dx**2 + dy**2)

        if dist <= step_size:
            return to_point

        ratio = step_size / dist
        new_x = int(from_point[0] + dx * ratio)
        new_y = int(from_point[1] + dy * ratio)

        # Ensure within bounds
        new_x = max(0, min(self.map_width - 1, new_x))
        new_y = max(0, min(self.map_height - 1, new_y))

        return (new_x, new_y)

    def is_valid_connection(self, p1, p2):
        """
        Check if connection between two points is valid (no obstacles).
        """
        # Use Bresenham's line algorithm to check all cells along the line
        points = self.bresenham_line(p1[0], p1[1], p2[0], p2[1])

        for x, y in points:
            if not self.is_valid_cell(x, y):
                return False

        return True

    def bresenham_line(self, x0, y0, x1, y1):
        """
        Bresenham's line algorithm to get all points between two points.
        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1
        error = dx - dy

        x, y = x0, y0

        while True:
            points.append((x, y))

            if x == x1 and y == y1:
                break

            error2 = 2 * error
            if error2 > -dy:
                error -= dy
                x += x_step
            if error2 < dx:
                error += dx
                y += y_step

        return points

    def is_valid_cell(self, x, y):
        """
        Check if a grid cell is valid (not an obstacle).
        """
        if x < 0 or x >= self.map_width or y < 0 or y >= self.map_height:
            return False

        # Check occupancy value (0 = free, 100 = occupied)
        occupancy = self.map_data[y, x]
        return occupancy < 50  # Consider cells with <50% occupancy as free

    def distance(self, p1, p2):
        """
        Calculate Euclidean distance between two points.
        """
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def reconstruct_path(self, tree, start, goal):
        """
        Reconstruct path from tree.
        """
        path = [goal]
        current = goal

        while current != start:
            parent = tree[current]
            path.append(parent)
            current = parent

        path.reverse()
        return path

    def rewire_tree(self, tree, costs, new_point, step_size):
        """
        Rewire the RRT* tree to improve path quality.
        """
        # Find all nodes within rewire radius
        radius = min(50, 2 * step_size)  # Adjust radius as needed

        for node in tree:
            if node == new_point:
                continue

            if self.distance(node, new_point) <= radius:
                # Check if rewiring improves cost
                new_cost = costs[new_point] + self.distance(new_point, node)

                if new_cost < costs.get(node, float('inf')):
                    if self.is_valid_connection(new_point, node):
                        tree[node] = new_point
                        costs[node] = new_cost

    def smooth_humonoid_path(self, path):
        """
        Smooth path considering humanoid locomotion constraints.
        """
        if len(path) < 3:
            return path

        smoothed = [path[0]]
        i = 0

        while i < len(path) - 1:
            j = len(path) - 1

            # Try to connect current point to furthest possible point
            while j > i:
                if self.is_valid_humonoid_connection(path[i], path[j]):
                    smoothed.append(path[j])
                    i = j
                    break
                j -= 1

            if j == i:  # No valid connection found, advance by one
                i += 1
                if i < len(path):
                    smoothed.append(path[i])

        return smoothed

    def is_valid_humonoid_connection(self, p1, p2):
        """
        Check if connection between two points is valid for humanoid.
        Considers step size and angle constraints.
        """
        # Check distance constraint
        dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        if dist * self.map_resolution > 0.8:  # Max step distance
            return False

        # Check if path between points is collision-free
        return self.is_valid_connection(p1, p2)

    def generate_humonoid_trajectory(self, path):
        """
        Generate a trajectory considering humanoid dynamics and footstep planning.
        """
        if len(path) < 2:
            return []

        trajectory = []
        current_time = self.get_clock().now()

        # Add start point
        start_pose = PoseStamped()
        start_pose.header.stamp = current_time.to_msg()
        start_pose.header.frame_id = 'map'
        start_pose.pose.position.x = path[0][0]
        start_pose.pose.position.y = path[0][1]
        start_pose.pose.position.z = 0.0  # Ground level
        trajectory.append(start_pose)

        # Generate intermediate points
        for i in range(1, len(path)):
            # Calculate intermediate poses along the segment
            num_intermediate = max(1, int(self.distance((path[i-1]), (path[i])) / 0.1))

            for j in range(1, num_intermediate + 1):
                ratio = j / num_intermediate
                x = path[i-1][0] + (path[i][0] - path[i-1][0]) * ratio
                y = path[i-1][1] + (path[i][1] - path[i-1][1]) * ratio

                pose = PoseStamped()
                pose.header.stamp = (current_time + rclpy.time.Duration(seconds=i*0.1)).to_msg()
                pose.header.frame_id = 'map'
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.0  # Ground level

                # Calculate orientation to face direction of movement
                if i > 0:
                    dx = path[i][0] - path[i-1][0]
                    dy = path[i][1] - path[i-1][1]
                    yaw = np.arctan2(dy, dx)
                    quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
                    pose.pose.orientation.x = quat[0]
                    pose.pose.orientation.y = quat[1]
                    pose.pose.orientation.z = quat[2]
                    pose.pose.orientation.w = quat[3]

                trajectory.append(pose)

        return trajectory

    def world_to_grid(self, x, y):
        """
        Convert world coordinates to grid coordinates.
        """
        grid_x = int((x - self.map_origin[0]) / self.map_resolution)
        grid_y = int((y - self.map_origin[1]) / self.map_resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid coordinates to world coordinates.
        """
        x = grid_x * self.map_resolution + self.map_origin[0]
        y = grid_y * self.map_resolution + self.map_origin[1]
        return (x, y)

    def publish_path(self, path_points):
        """
        Publish the planned path.
        """
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for x, y in path_points:
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.get_logger().info(f'Published path with {len(path_points)} waypoints')

    def publish_trajectory(self, trajectory):
        """
        Publish the generated trajectory.
        """
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        path_msg.poses = trajectory

        self.trajectory_pub.publish(path_msg)
        self.get_logger().info(f'Published trajectory with {len(trajectory)} points')

    def visualize_path(self, path_points, trajectory):
        """
        Visualize the path and trajectory using markers.
        """
        marker_array = MarkerArray()

        # Path visualization
        path_marker = Marker()
        path_marker.header.frame_id = 'map'
        path_marker.header.stamp = self.get_clock().now().to_msg()
        path_marker.ns = 'path'
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.05  # Line width
        path_marker.color.r = 0.0
        path_marker.color.g = 1.0
        path_marker.color.b = 0.0
        path_marker.color.a = 1.0

        for x, y in path_points:
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.05  # Slightly above ground for visibility
            path_marker.points.append(point)

        marker_array.markers.append(path_marker)

        # Trajectory visualization
        traj_marker = Marker()
        traj_marker.header.frame_id = 'map'
        traj_marker.header.stamp = self.get_clock().now().to_msg()
        traj_marker.ns = 'trajectory'
        traj_marker.id = 1
        traj_marker.type = Marker.LINE_STRIP
        traj_marker.action = Marker.ADD
        traj_marker.scale.x = 0.02  # Line width
        traj_marker.color.r = 1.0
        traj_marker.color.g = 0.0
        traj_marker.color.b = 0.0
        traj_marker.color.a = 1.0

        for pose_stamped in trajectory:
            point = Point()
            point.x = pose_stamped.pose.position.x
            point.y = pose_stamped.pose.position.y
            point.z = 0.1  # Slightly above ground for visibility
            traj_marker.points.append(point)

        marker_array.markers.append(traj_marker)

        # Goal marker
        goal_marker = Marker()
        goal_marker.header.frame_id = 'map'
        goal_marker.header.stamp = self.get_clock().now().to_msg()
        goal_marker.ns = 'goal'
        goal_marker.id = 2
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.scale.x = 0.3
        goal_marker.scale.y = 0.3
        goal_marker.scale.z = 0.3
        goal_marker.color.r = 1.0
        goal_marker.color.g = 1.0
        goal_marker.color.b = 0.0
        goal_marker.color.a = 1.0
        goal_marker.pose.position.x = self.goal_pose.position.x
        goal_marker.pose.position.y = self.goal_pose.position.y
        goal_marker.pose.position.z = 0.2

        marker_array.markers.append(goal_marker)

        self.visualization_pub.publish(marker_array)


class IsaacROSPathPlannerInterface(Node):
    """
    Interface between Isaac Sim path planning and ROS 2.
    """

    def __init__(self):
        super().__init__('isaac_ros_path_planner_interface')

        # Publishers for Isaac Sim
        self.footstep_pub = self.create_publisher(Path, '/humanoid_footsteps', 10)
        self.com_trajectory_pub = self.create_publisher(Path, '/humanoid_com_trajectory', 10)

        # Subscribers for robot state
        self.robot_state_sub = self.create_subscription(
            Path,  # This would be a custom message in practice
            '/humanoid_state',
            self.robot_state_callback,
            10
        )

        # Timer for periodic processing
        self.process_timer = self.create_timer(0.1, self.process_path_planning)

        # Path planning components
        self.path_planner = HumanoidPathPlanner()
        self.current_state = None
        self.planned_path = None

        self.get_logger().info('Isaac ROS Path Planner Interface initialized')

    def robot_state_callback(self, msg):
        """
        Process robot state messages from Isaac Sim.
        """
        # Update current robot state
        self.current_state = msg

    def process_path_planning(self):
        """
        Process path planning based on current state and goals.
        """
        # This would integrate with the path planner to handle dynamic replanning
        pass


def main(args=None):
    rclpy.init(args=args)

    # Create path planner node
    path_planner = HumanoidPathPlanner()

    try:
        rclpy.spin(path_planner)
    except KeyboardInterrupt:
        path_planner.get_logger().info('Shutting down path planner...')
    finally:
        path_planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
# Sensor Simulation: LiDAR, Depth Cameras, IMUs

## Introduction

Sensor simulation is a critical aspect of robotics development, allowing developers to test perception algorithms, navigation systems, and control strategies without physical hardware. This chapter covers the simulation of three key sensor types in Gazebo: LiDAR, depth cameras, and IMUs, which are essential for robot perception and navigation.

## LiDAR Simulation

### Types of LiDAR Sensors

Gazebo supports several types of LiDAR sensors:
- **Ray sensors**: Basic 2D laser range finders
- **GPU ray sensors**: GPU-accelerated 2D laser range finders
- **3D ray sensors**: Multi-line LiDAR (e.g., Velodyne-style)

### 2D LiDAR Configuration

```xml
<gazebo reference="laser_link">
  <sensor name="laser_sensor" type="ray">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>40</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-1.570796</min_angle>  <!-- -90 degrees -->
          <max_angle>1.570796</max_angle>   <!-- 90 degrees -->
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/laser</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>
```

### 3D LiDAR Configuration

```xml
<gazebo reference="velodyne_link">
  <sensor name="velodyne_sensor" type="ray">
    <pose>0 0 0 0 0 0</pose>
    <visualize>false</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>1800</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
        <vertical>
          <samples>32</samples>
          <resolution>1</resolution>
          <min_angle>-0.436332</min_angle>  <!-- -25 degrees -->
          <max_angle>0.279253</max_angle>   <!-- 16 degrees -->
        </vertical>
      </scan>
      <range>
        <min>0.2</min>
        <max>100.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="velodyne_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/velodyne</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/PointCloud2</output_type>
    </plugin>
  </sensor>
</gazebo>
```

## Depth Camera Simulation

### Depth Camera Configuration

```xml
<gazebo reference="camera_link">
  <sensor name="depth_camera" type="depth">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>30</update_rate>
    <camera name="depth_camera">
      <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/camera</namespace>
        <remapping>image_raw:=image</remapping>
        <remapping>camera_info:=camera_info</remapping>
      </ros>
      <camera_name>depth_camera</camera_name>
      <frame_name>camera_optical_frame</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>10.0</max_depth>
    </plugin>
  </sensor>
</gazebo>
```

### RGB-D Camera with Point Cloud Output

```xml
<gazebo reference="rgbd_camera_link">
  <sensor name="rgbd_camera" type="depth">
    <always_on>true</always_on>
    <visualize>true</visualize>
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <baseline>0.2</baseline>
      <alwaysOn>true</alwaysOn>
      <updateRate>30.0</updateRate>
      <cameraName>rgbd_camera</cameraName>
      <imageTopicName>/rgb/image_raw</imageTopicName>
      <depthImageTopicName>/depth/image_raw</depthImageTopicName>
      <pointCloudTopicName>/depth/points</pointCloudTopicName>
      <cameraInfoTopicName>/rgb/camera_info</cameraInfoTopicName>
      <depthImageCameraInfoTopicName>/depth/camera_info</depthImageCameraInfoTopicName>
      <frameName>rgbd_camera_optical_frame</frameName>
      <pointCloudCutoff>0.1</pointCloudCutoff>
      <pointCloudCutoffMax>10.0</pointCloudCutoffMax>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
      <CxPrime>0.0</CxPrime>
      <Cx>0.0</Cx>
      <Cy>0.0</Cy>
      <focalLength>0.0</focalLength>
      <hackBaseline>0.0</hackBaseline>
    </plugin>
  </sensor>
</gazebo>
```

## IMU Simulation

### IMU Sensor Configuration

```xml
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>  <!-- ~0.1 deg/s (1-sigma) -->
            <bias_mean>0.005</bias_mean>
            <bias_stddev>0.0005</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>
            <bias_mean>0.005</bias_mean>
            <bias_stddev>0.0005</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>
            <bias_mean>0.005</bias_mean>
            <bias_stddev>0.0005</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>  <!-- 1-sigma accel noise: 17 mg */
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>/imu</namespace>
        <remapping>~/out:=data</remapping>
      </ros>
      <frame_name>imu_link</frame_name>
      <body_name>imu_link</body_name>
      <update_rate>100</update_rate>
      <gaussian_noise>0.0017</gaussian_noise>
      <accel_gaussian_noise>0.017</accel_gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

## Working with Sensor Data in ROS 2

### LiDAR Data Processing

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

class LIDARProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')

        self.subscription = self.create_subscription(
            LaserScan,
            '/laser/scan',
            self.lidar_callback,
            10
        )

        self.get_logger().info('LiDAR Processor initialized')

    def lidar_callback(self, msg):
        # Convert to numpy array for processing
        ranges = np.array(msg.ranges)

        # Filter out invalid ranges (inf, nan)
        valid_ranges = ranges[np.isfinite(ranges)]

        # Find minimum distance (obstacle detection)
        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            min_idx = np.argmin(ranges)
            angle_of_min = msg.angle_min + min_idx * msg.angle_increment

            self.get_logger().info(
                f'Min distance: {min_distance:.2f}m at angle: {angle_of_min:.2f}rad'
            )

        # Calculate statistics
        if len(valid_ranges) > 0:
            avg_distance = np.mean(valid_ranges)
            self.get_logger().info(f'Average distance: {avg_distance:.2f}m')
```

### Depth Camera Data Processing

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthCameraProcessor(Node):
    def __init__(self):
        super().__init__('depth_camera_processor')

        self.bridge = CvBridge()

        # Subscribe to depth image
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        # Subscribe to camera info for intrinsics
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self.info_callback,
            10
        )

        self.camera_info = None
        self.get_logger().info('Depth Camera Processor initialized')

    def info_callback(self, msg):
        self.camera_info = msg

    def depth_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Process depth data
            # Find distance to center pixel
            height, width = cv_image.shape
            center_x, center_y = width // 2, height // 2
            center_depth = cv_image[center_y, center_x]

            if np.isfinite(center_depth):
                self.get_logger().info(f'Depth at center: {center_depth:.2f}m')

            # Find valid depth pixels (not inf or nan)
            valid_depths = cv_image[np.isfinite(cv_image)]
            if len(valid_depths) > 0:
                avg_depth = np.mean(valid_depths)
                self.get_logger().info(f'Average depth: {avg_depth:.2f}m')

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')
```

### IMU Data Processing

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import math

class IMUProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')

        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.get_logger().info('IMU Processor initialized')

    def imu_callback(self, msg):
        # Extract orientation (quaternion)
        orientation = msg.orientation
        # Convert quaternion to Euler angles
        euler = self.quaternion_to_euler(orientation)

        # Extract angular velocity
        angular_vel = msg.angular_velocity

        # Extract linear acceleration
        linear_acc = msg.linear_acceleration

        self.get_logger().info(
            f'Roll: {euler.x:.2f}, Pitch: {euler.y:.2f}, Yaw: {euler.z:.2f}'
        )
        self.get_logger().info(
            f'Angular Vel - X: {angular_vel.x:.2f}, Y: {angular_vel.y:.2f}, Z: {angular_vel.z:.2f}'
        )
        self.get_logger().info(
            f'Linear Acc - X: {linear_acc.x:.2f}, Y: {linear_acc.y:.2f}, Z: {linear_acc.z:.2f}'
        )

    def quaternion_to_euler(self, quaternion):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        q = quaternion
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(sinp)

        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return Vector3(x=roll, y=pitch, z=yaw)
```

## Sensor Fusion Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Subscribers for different sensors
        self.lidar_sub = self.create_subscription(
            LaserScan, '/laser/scan', self.lidar_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # Publisher for fused pose
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/fused_pose', 10
        )

        # Initialize state variables
        self.orientation = [0.0, 0.0, 0.0, 1.0]  # x, y, z, w
        self.position = [0.0, 0.0, 0.0]

        self.get_logger().info('Sensor Fusion Node initialized')

    def lidar_callback(self, msg):
        # Process LiDAR data for position estimation
        # This is a simplified example - real fusion would be more complex
        pass

    def imu_callback(self, msg):
        # Update orientation from IMU
        self.orientation = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]

    def publish_fused_pose(self):
        # Publish the fused pose estimate
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        # Set position and orientation
        pose_msg.pose.pose.position.x = self.position[0]
        pose_msg.pose.pose.position.y = self.position[1]
        pose_msg.pose.pose.position.z = self.position[2]

        pose_msg.pose.pose.orientation.x = self.orientation[0]
        pose_msg.pose.pose.orientation.y = self.orientation[1]
        pose_msg.pose.pose.orientation.z = self.orientation[2]
        pose_msg.pose.pose.orientation.w = self.orientation[3]

        # Set covariance (simplified)
        pose_msg.pose.covariance = [0.1] * 36  # Diagonal elements = 0.1

        self.pose_pub.publish(pose_msg)
```

## Best Practices for Sensor Simulation

### LiDAR Best Practices

1. **Resolution**: Balance between detail and performance
2. **Range**: Set appropriate min/max values for your application
3. **Update rate**: Match the rate to your algorithm requirements
4. **Noise**: Include realistic noise models for robust algorithms
5. **Field of view**: Match the physical sensor specifications

### Camera Best Practices

1. **Intrinsics**: Use accurate camera calibration parameters
2. **Distortion**: Include distortion coefficients if present in real sensor
3. **Frame rate**: Match the physical camera capabilities
4. **Resolution**: Use appropriate resolution for your algorithms
5. **Noise**: Add realistic noise models for robust processing

### IMU Best Practices

1. **Bias**: Include realistic bias values
2. **Noise**: Use appropriate noise models based on sensor specifications
3. **Update rate**: Higher rates for dynamic applications
4. **Alignment**: Ensure proper frame alignment with robot coordinate system
5. **Calibration**: Simulate calibration procedures in your algorithms

### General Best Practices

1. **Validate with real data**: Compare simulation output with real sensor data
2. **Document parameters**: Keep detailed records of sensor configurations
3. **Test edge cases**: Include scenarios with sensor limitations
4. **Performance monitoring**: Monitor simulation performance with multiple sensors
5. **Cross-validation**: Test algorithms with different sensor combinations
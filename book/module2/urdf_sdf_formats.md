# URDF and SDF Robot Description Formats

## Introduction

URDF (Unified Robot Description Format) and SDF (Simulation Description Format) are two XML-based formats used to describe robots in ROS and Gazebo respectively. Understanding both formats is essential for creating robots that work seamlessly in both ROS environments and Gazebo simulations.

## URDF Overview

URDF is primarily used in ROS for representing robot models. It describes the kinematic and dynamic properties of a robot, including links, joints, and visual/collision properties.

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="joint_name" type="revolute">
    <parent link="base_link"/>
    <child link="child_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>

  <link name="child_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.5"/>
      </geometry>
    </visual>
  </link>
</robot>
```

## SDF Overview

SDF is used by Gazebo and supports more features than URDF, including plugins, worlds, and simulation-specific properties.

### Basic SDF Structure

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="my_robot">
    <link name="base_link">
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.1</iyy>
          <iyz>0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry>
          <box>
            <size>1 1 1</size>
          </box>
        </geometry>
      </visual>
      <collision name="collision">
        <geometry>
          <box>
            <size>1 1 1</size>
          </box>
        </geometry>
      </collision>
    </link>

    <joint name="joint_name" type="revolute">
      <parent>base_link</parent>
      <child>child_link</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>10</effort>
          <velocity>1</velocity>
        </limit>
      </axis>
    </joint>

    <link name="child_link">
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.5</length>
          </cylinder>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
```

## Converting URDF to SDF

### Using xacro with Gazebo

For robots defined in URDF, you can add Gazebo-specific extensions:

```xml
<!-- Include in URDF file -->
<gazebo reference="link_name">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
</gazebo>

<!-- For joints -->
<gazebo reference="joint_name">
  <provideFeedback>true</provideFeedback>
  <implicitSpringDamper>1</implicitSpringDamper>
</gazebo>

<!-- Adding transmission for control -->
<xacro:macro name="transmission_block" params="joint_name">
  <transmission name="${joint_name}_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="${joint_name}">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="${joint_name}_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>
</xacro:macro>
```

### Converting Command

```bash
# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf

# Or using the older command
gzsdf -p robot.urdf > robot.sdf
```

## Gazebo-Specific Extensions in URDF

### Adding Sensors

```xml
<gazebo reference="sensor_link">
  <sensor name="camera" type="camera">
    <pose>0 0 0 0 0 0</pose>
    <camera name="camera">
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
    <always_on>1</always_on>
    <update_rate>30</update_rate>
    <visualize>true</visualize>
  </sensor>
</gazebo>
```

### Adding Physics Properties

```xml
<gazebo reference="link_name">
  <mu1>0.9</mu1>
  <mu2>0.9</mu2>
  <kp>1e+9</kp>
  <kd>1e+6</kd>
  <max_vel>100.0</max_vel>
  <min_depth>0.001</min_depth>
  <fdir1>1 0 0</fdir1>
</gazebo>
```

### Adding Plugins

```xml
<gazebo>
  <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
    <ros>
      <namespace>demo</namespace>
      <remapping>cmd_vel:=cmd_demo</remapping>
      <remapping>odom:=odom_demo</remapping>
    </ros>
    <update_rate>30</update_rate>
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.3</wheel_separation>
    <wheel_diameter>0.15</wheel_diameter>
    <max_wheel_torque>20</max_wheel_torque>
    <max_wheel_acceleration>1.0</max_wheel_acceleration>
    <command_topic>cmd_vel</command_topic>
    <odometry_topic>odom</odometry_topic>
    <odometry_frame>odom</odometry_frame>
    <robot_base_frame>base_footprint</robot_base_frame>
    <publish_odom>true</publish_odom>
    <publish_wheel_tf>false</publish_wheel_tf>
    <publish_odom_tf>true</publish_odom_tf>
    <odometry_source>world</odometry_source>
  </plugin>
</gazebo>
```

## Working with Xacro

Xacro is a macro language for XML that makes URDF files more readable and maintainable:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_robot">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.05" />
  <xacro:property name="wheel_width" value="0.02" />

  <!-- Macro for wheels -->
  <xacro:macro name="wheel" params="prefix parent xyz rpy">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
    </joint>

    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.2"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </visual>
  </link>

  <!-- Use the macro to create wheels -->
  <xacro:wheel prefix="front_left" parent="base_link" xyz="0.15 0.15 0" rpy="0 0 0"/>
  <xacro:wheel prefix="front_right" parent="base_link" xyz="0.15 -0.15 0" rpy="0 0 0"/>
  <xacro:wheel prefix="back_left" parent="base_link" xyz="-0.15 0.15 0" rpy="0 0 0"/>
  <xacro:wheel prefix="back_right" parent="base_link" xyz="-0.15 -0.15 0" rpy="0 0 0"/>

</robot>
```

## SDF Worlds

SDF can also describe entire worlds with multiple models:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="my_world">
    <!-- Include models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Direct model definition -->
    <model name="my_robot">
      <!-- Robot definition here -->
    </model>

    <!-- Static objects -->
    <model name="table">
      <pose>1 1 0 0 0 0</pose>
      <link name="table_base">
        <visual name="visual">
          <geometry>
            <box>
              <size>1 0.8 0.8</size>
            </box>
          </geometry>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>1 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
      </link>
    </model>

    <!-- Physics properties -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
  </world>
</sdf>
```

## Best Practices

### URDF Best Practices

1. **Use consistent naming**: Follow ROS naming conventions (snake_case)
2. **Start simple**: Begin with basic shapes, add detail gradually
3. **Validate**: Use `check_urdf` command to validate your URDF
4. **Use xacro**: For complex robots, use xacro to avoid repetition
5. **Include inertial properties**: Essential for physics simulation
6. **Test in RViz first**: Verify kinematic structure before simulation

### SDF Best Practices

1. **Use appropriate SDF version**: Match your Gazebo version
2. **Include proper mass properties**: Essential for stable simulation
3. **Add collision geometry**: Separate from visual geometry if needed
4. **Use plugins wisely**: Only add plugins that are necessary
5. **Test physics properties**: Validate with simple tests before complex scenarios

### Conversion Best Practices

1. **Test both formats**: Ensure the robot works in both URDF and SDF
2. **Validate transforms**: Check that all frames are properly connected
3. **Check joint limits**: Ensure they're appropriate for your application
4. **Verify sensor placement**: Ensure sensors are positioned correctly
5. **Performance testing**: Monitor simulation performance with complex models

## Tools for Working with URDF/SDF

### Validation Tools

```bash
# Validate URDF
check_urdf robot.urdf

# Convert and check SDF
gz sdf -p robot.urdf

# Visualize URDF in RViz
ros2 run rviz2 rviz2
```

### Visualization Tools

```bash
# Visualize URDF
ros2 run xacro xacro --inorder robot.urdf | gz sdf -p

# Check joint limits
ros2 run urdf_parser joint_limits_aggregator robot.urdf
```
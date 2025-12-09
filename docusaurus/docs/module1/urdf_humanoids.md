# URDF for Humanoid Robots

## Introduction to URDF

URDF (Unified Robot Description Format) is an XML-based format used to describe robot models in ROS. It defines the physical and visual properties of a robot, including its links, joints, inertial properties, and visual appearance. URDF is essential for simulation, visualization, and kinematic analysis of robots.

## URDF Structure

A URDF file consists of:
- **Links**: Rigid bodies that make up the robot
- **Joints**: Connections between links that allow relative motion
- **Visual**: How the link appears in visualization
- **Collision**: Collision properties for physics simulation
- **Inertial**: Mass properties for physics simulation

## Basic URDF Example

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.083" ixy="0" ixz="0" iyy="0.083" iyz="0" izz="0.125"/>
    </inertial>
  </link>

  <!-- Leg link -->
  <link name="leg">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.05"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
  </link>

  <!-- Joint connecting base to leg -->
  <joint name="base_to_leg" type="fixed">
    <parent link="base_link"/>
    <child link="leg"/>
    <origin xyz="0 0 -0.3"/>
  </joint>
</robot>
```

## Links

Links represent rigid bodies in the robot. Each link contains:
- **Visual**: How the link looks in RViz and simulators
- **Collision**: How the link behaves in physics simulations
- **Inertial**: Mass, center of mass, and inertia tensor

### Visual Properties
```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <!-- box, cylinder, sphere, or mesh -->
    <box size="1 1 1"/>
  </geometry>
  <material name="red">
    <color rgba="1 0 0 1"/>
  </material>
</visual>
```

### Collision Properties
```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <box size="1 1 1"/>
  </geometry>
</collision>
```

### Inertial Properties
```xml
<inertial>
  <mass value="1.0"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <inertia ixx="0.083" ixy="0" ixz="0" iyy="0.083" iyz="0" izz="0.083"/>
</inertial>
```

## Joints

Joints connect links and define how they can move relative to each other. Joint types include:
- **revolute**: Rotational joint with limited range
- **continuous**: Rotational joint without limits
- **prismatic**: Linear sliding joint with limits
- **fixed**: No movement between links
- **floating**: 6 DOF movement
- **planar**: Movement on a plane

### Joint Example
```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
```

## Humanoid Robot Structure

A humanoid robot typically has the following structure:
- **Torso/Body**: Main body containing computational units
- **Head**: Contains cameras, sensors, and processing units
- **Arms**: With shoulder, elbow, and wrist joints
- **Legs**: With hip, knee, and ankle joints
- **Hands**: For manipulation tasks

### Example Humanoid Structure
```xml
<robot name="humanoid_robot">
  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </visual>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </visual>
  </link>

  <joint name="torso_to_head" type="fixed">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.35"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </visual>
  </link>

  <joint name="left_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0 0.2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
</robot>
```

## Xacro for Complex URDFs

Xacro (XML Macros) allows you to use variables, macros, and mathematical expressions in URDF files, making them more readable and maintainable:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="torso_height" value="0.5" />

  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 ${torso_height}"/>
      </geometry>
    </visual>
  </link>

  <xacro:macro name="arm" params="side">
    <link name="${side}_upper_arm">
      <visual>
        <geometry>
          <cylinder length="0.3" radius="0.05"/>
        </geometry>
      </visual>
    </link>
  </xacro:macro>

  <xacro:arm side="left"/>
  <xacro:arm side="right"/>
</robot>
```

## Best Practices

1. Start with a simple model and gradually add complexity
2. Use consistent naming conventions
3. Ensure all links are connected through joints
4. Include proper inertial properties for physics simulation
5. Use Xacro for complex robots to avoid repetition
6. Validate URDF files using `check_urdf` command
7. Test the kinematic chain with `rviz`
8. Consider using standard dimensions for humanoid proportions
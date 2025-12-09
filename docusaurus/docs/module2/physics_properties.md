# Physics Properties: Gravity, Collisions, and More

## Introduction

Understanding physics properties is crucial for creating realistic robot simulations in Gazebo. This chapter covers the fundamental physics concepts that govern how robots and objects behave in simulation, including gravity, collisions, friction, and other physical properties that affect robot motion and interaction.

## Gravity in Gazebo

### Setting Global Gravity

Gravity is defined globally for each world and affects all objects in the simulation:

```xml
<sdf version="1.7">
  <world name="my_world">
    <!-- Set gravity vector (x, y, z) in m/s^2 -->
    <gravity>0 0 -9.8</gravity>
    <!-- ... rest of world definition -->
  </world>
</sdf>
```

### Gravity Considerations

- Standard Earth gravity: 9.8 m/s² downward (0 0 -9.8)
- Different celestial bodies have different gravity values
- Zero gravity can be useful for testing floating robots
- Direction matters: negative Z is typically "down"

## Collision Properties

### Collision Detection

Collision properties define how objects interact when they come into contact:

```xml
<link name="link_name">
  <collision name="collision_name">
    <geometry>
      <box>
        <size>1 1 1</size>
      </box>
    </geometry>
    <!-- Surface properties -->
    <surface>
      <friction>
        <ode>
          <mu>1.0</mu>
          <mu2>1.0</mu2>
        </ode>
      </friction>
      <bounce>
        <restitution_coefficient>0.1</restitution_coefficient>
        <threshold>100000</threshold>
      </bounce>
      <contact>
        <ode>
          <kp>1e+16</kp>
          <kd>1e+13</kd>
          <max_vel>100.0</max_vel>
          <min_depth>0.001</min_depth>
        </ode>
      </contact>
    </surface>
  </collision>
</link>
```

### Collision Geometry Types

1. **Box**: Rectangular prism
2. **Cylinder**: Cylindrical shape
3. **Sphere**: Spherical shape
4. **Mesh**: Complex shapes from 3D models
5. **Plane**: Infinite flat surface

## Inertial Properties

### Mass and Center of Mass

Inertial properties determine how objects respond to forces:

```xml
<link name="link_name">
  <inertial>
    <!-- Mass in kilograms -->
    <mass>1.0</mass>
    <!-- Center of mass offset -->
    <pose>0.1 0 0 0 0 0</pose>
    <!-- Inertia matrix -->
    <inertia>
      <ixx>0.083</ixx>
      <ixy>0</ixy>
      <ixz>0</ixz>
      <iyy>0.083</iyy>
      <iyz>0</iyz>
      <izz>0.167</izz>
    </inertia>
  </inertial>
</link>
```

### Calculating Inertial Properties

For common shapes with uniform density ρ:

**Box** (width w, depth d, height h):
- Mass: m = ρ × w × d × h
- Ixx = (1/12) × m × (d² + h²)
- Iyy = (1/12) × m × (w² + h²)
- Izz = (1/12) × m × (w² + d²)

**Cylinder** (radius r, height h):
- Mass: m = ρ × π × r² × h
- Ixx = Iyy = (1/12) × m × (3r² + h²)
- Izz = (1/2) × m × r²

**Sphere** (radius r):
- Mass: m = ρ × (4/3) × π × r³
- Ixx = Iyy = Izz = (2/5) × m × r²

## Friction Models

### ODE (Open Dynamics Engine) Friction

Gazebo uses ODE for physics simulation. Friction parameters include:

- **μ (mu)**: Primary friction coefficient (longitudinal)
- **μ₂ (mu2)**: Secondary friction coefficient (lateral)
- **slip1**: Inverse of longitudinal slip
- **slip2**: Inverse of lateral slip

```xml
<surface>
  <friction>
    <ode>
      <mu>0.5</mu>
      <mu2>0.5</mu2>
      <slip1>0</slip1>
      <slip2>0</slip2>
    </ode>
  </friction>
</surface>
```

### Typical Friction Coefficients

| Material Combination | Static Friction μ |
|---------------------|------------------|
| Rubber on dry concrete | 1.0 |
| Steel on steel | 0.74 |
| Wood on wood | 0.25 - 0.5 |
| Ice on ice | 0.1 |
| Teflon on Teflon | 0.04 |

## Bounce and Restitution

### Restitution Coefficient

The restitution coefficient determines how "bouncy" collisions are:

```xml
<surface>
  <bounce>
    <!-- 0.0 = no bounce (perfectly inelastic) -->
    <!-- 1.0 = perfect bounce (elastic) -->
    <restitution_coefficient>0.2</restitution_coefficient>
    <!-- Energy threshold for bounce -->
    <threshold>100000</threshold>
  </bounce>
</surface>
```

## Contact Properties

### Contact Stiffness and Damping

Contact properties determine how objects respond when they touch:

```xml
<surface>
  <contact>
    <ode>
      <!-- Spring stiffness (penetration resistance) -->
      <kp>1e+16</kp>
      <!-- Damping coefficient -->
      <kd>1e+13</kd>
      <!-- Maximum contact penetration velocity -->
      <max_vel>100.0</max_vel>
      <!-- Minimum contact depth -->
      <min_depth>0.001</min_depth>
    </ode>
  </contact>
</surface>
```

## Physics Performance Tuning

### Time Step and Iterations

Adjust physics parameters for simulation stability vs. performance:

```xml
<world name="my_world">
  <physics type="ode">
    <!-- Physics engine parameters -->
    <max_step_size>0.001</max_step_size>  <!-- Time step (smaller = more accurate) -->
    <real_time_factor>1.0</real_time_factor>  <!-- Target simulation speed -->
    <real_time_update_rate>1000</real_time_update_rate>  <!-- Hz -->
    <ode>
      <!-- Solver iterations (more = more stable but slower) -->
      <solver>
        <type>quick</type>
        <iters>100</iters>
        <sor>1.3</sor>
      </solver>
      <!-- Constraint parameters -->
      <constraints>
        <cfm>0.0</cfm>
        <erp>0.2</erp>
        <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
        <contact_surface_layer>0.001</contact_surface_layer>
      </constraints>
    </ode>
  </physics>
</world>
```

## Robot-Specific Physics Considerations

### Joint Dynamics

For realistic joint behavior, include dynamic properties:

```xml
<joint name="joint_name" type="revolute">
  <parent>parent_link</parent>
  <child>child_link</child>
  <axis>
    <xyz>0 0 1</xyz>
    <dynamics>
      <!-- Damping coefficient -->
      <damping>0.1</damping>
      <!-- Friction coefficient -->
      <friction>0.0</friction>
      <!-- Spring stiffness -->
      <spring_reference>0.0</spring_reference>
      <spring_stiffness>0.0</spring_stiffness>
    </dynamics>
    <limit>
      <lower>-1.57</lower>
      <upper>1.57</upper>
      <effort>100</effort>
      <velocity>1</velocity>
    </limit>
  </axis>
</joint>
```

### Actuator Simulation

Simulate actuator behavior with realistic dynamics:

```xml
<gazebo reference="joint_name">
  <provideFeedback>true</provideFeedback>
  <joint_properties>
    <ode_joint_config>
      <fudge_factor>1.0</fudge_factor>
      <cfm_damping>1</cfm_damping>
      <implicit_spring_damper>1</implicit_spring_damper>
    </ode_joint_config>
  </joint_properties>
</gazebo>
```

## Troubleshooting Physics Issues

### Common Problems and Solutions

1. **Objects falling through surfaces**: Increase contact stiffness (kp) or decrease min_depth
2. **Unstable simulations**: Reduce time step or increase solver iterations
3. **Objects sliding unrealistically**: Increase friction coefficients
4. **Objects bouncing too much**: Decrease restitution coefficient
5. **Robot joints acting stiffly**: Adjust joint damping values

## Best Practices

1. **Use realistic values**: Base physics parameters on real-world measurements
2. **Start conservative**: Use stable values first, then optimize for performance
3. **Test incrementally**: Add physics properties one at a time
4. **Validate with reality**: Compare simulation behavior with real robots when possible
5. **Document parameters**: Keep a record of physics values used for reproducibility
6. **Consider computational cost**: Balance accuracy with simulation performance
7. **Use appropriate geometry**: Match collision geometry to visual geometry for consistency
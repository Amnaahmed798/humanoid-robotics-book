# Unity Visualization for Robots

## Introduction

While Gazebo provides excellent physics simulation, Unity offers high-quality visualization and rendering capabilities that can enhance the robot development experience. Unity's advanced graphics, real-time rendering, and user interaction capabilities make it an excellent complement to physics-based simulators like Gazebo.

## Unity for Robotics Overview

Unity provides the Unity Robotics Hub, which includes tools and packages specifically designed for robotics development:
- **Unity Robotics Package**: Tools for ROS communication
- **Unity Perception Package**: Tools for generating synthetic training data
- **Unity Simulation Package**: Tools for distributed simulation

## Setting up Unity for Robotics

### Installing Unity

1. Download Unity Hub from the Unity website
2. Install Unity Editor (2021.3 LTS or later recommended)
3. Create a Unity account if you don't have one

### Installing Robotics Packages

In Unity Package Manager (Window > Package Manager):
1. Install "ROS-TCP-Connector" package
2. Install "ROS-TCP-Endpoint" package
3. Install "Unity Perception" package (optional)

## Basic Unity Scene Setup

### Creating a Robot Visualization Scene

```csharp
using UnityEngine;

public class RobotVisualizer : MonoBehaviour
{
    // Robot joint transforms
    public Transform baseLink;
    public Transform[] joints;

    // ROS communication
    private RosConnector rosConnector;

    void Start()
    {
        // Initialize ROS connection
        rosConnector = GetComponent<RosConnector>();
        rosConnector.rosSocket.ConnectToRos("127.0.0.1", 10000);

        // Subscribe to joint states
        rosConnector.rosSocket.Subscribe<sensor_msgs.JointState>(
            "/joint_states", JointStateCallback
        );
    }

    void JointStateCallback(sensor_msgs.JointState jointState)
    {
        // Update joint positions based on received joint states
        for (int i = 0; i < joints.Length && i < jointState.position.Count; i++)
        {
            // Apply joint position to Unity transform
            // This is a simplified example - actual implementation depends on joint type
            joints[i].localEulerAngles = new Vector3(
                0, 0, (float)(jointState.position[i] * Mathf.Rad2Deg)
            );
        }
    }
}
```

## ROS Communication in Unity

### Using ROS-TCP-Connector

The ROS-TCP-Connector package allows Unity to communicate with ROS networks:

```csharp
using ROS2;
using sensor_msgs;

public class UnityROSCommunicator : MonoBehaviour
{
    private ROS2UnityComponent ros2Unity;

    void Start()
    {
        ros2Unity = GetComponent<ROS2UnityComponent>();
        ros2Unity.Initialize();

        // Create publisher for visualization markers
        var markerPublisher = ros2Unity.CreatePublisher<visualization_msgs.Marker>(
            "/unity_visualization"
        );

        // Create subscriber for robot states
        var jointSubscriber = ros2Unity.CreateSubscription<sensor_msgs.JointState>(
            "/joint_states", JointStateCallback
        );
    }

    void JointStateCallback(JointState msg)
    {
        // Process joint state message
        // Update Unity objects based on joint positions
    }

    void OnDestroy()
    {
        ros2Unity.Shutdown();
    }
}
```

## Visualizing Robot Data

### Creating Visualization Markers

Unity can generate visualization markers similar to RViz:

```csharp
using visualization_msgs;
using geometry_msgs;

public class MarkerVisualizer : MonoBehaviour
{
    public void PublishMarker(string frameId, Vector3 position, Color color, string text = "")
    {
        var marker = new Marker();
        marker.header.frame_id = frameId;
        marker.header.stamp = new TimeStamp();
        marker.ns = "unity_markers";
        marker.id = 0;
        marker.type = Marker.TEXT_VIEW_FACING;
        marker.action = Marker.ADD;

        marker.pose = new Pose();
        marker.pose.position = new Vector3Msg(position.x, position.y, position.z);
        marker.pose.orientation = new QuaternionMsg(0, 0, 0, 1);

        marker.scale = new Vector3Msg(0.1f, 0.1f, 0.1f);
        marker.color = new ColorRGBA(color.r, color.g, color.b, color.a);
        marker.text = text;

        // Publish the marker
        // Implementation depends on your ROS communication setup
    }
}
```

## Unity Perception Package

The Unity Perception package enables synthetic data generation for machine learning:

### Synthetic Data Generation

```csharp
using UnityEngine;
using Unity.Perception.GroundTruth;
using Unity.Perception.Randomization.Scenarios;

public class PerceptionSetup : MonoBehaviour
{
    [SerializeField] private GameObject robot;
    [SerializeField] private Camera sensorCamera;

    void Start()
    {
        // Add semantic labels to robot parts
        AddSemanticLabels();

        // Configure camera for data collection
        ConfigureSensorCamera();
    }

    void AddSemanticLabels()
    {
        // Add semantic segmentation labels to robot parts
        var links = robot.GetComponentsInChildren<Renderer>();
        foreach (var link in links)
        {
            var semanticLabel = link.gameObject.AddComponent<SemanticSegmentationLabel>();
            semanticLabel.labelId = GetLabelId(link.name);
        }
    }

    void ConfigureSensorCamera()
    {
        // Add perception camera components
        var cameraSensor = sensorCamera.gameObject.AddComponent<CameraSensor>();
        cameraSensor.sensorId = "rgb_camera";

        // Add depth camera if needed
        var depthSensor = sensorCamera.gameObject.AddComponent<DepthSensor>();
        depthSensor.sensorId = "depth_camera";
    }

    int GetLabelId(string objectName)
    {
        // Map object names to semantic label IDs
        switch (objectName)
        {
            case "base_link": return 1;
            case "head": return 2;
            case "arm": return 3;
            default: return 0;
        }
    }
}
```

## Creating Robot Prefabs

### Robot Prefab Structure

A well-structured robot prefab in Unity should include:

```
RobotPrefab/
├── BaseLink/
│   ├── VisualMesh (MeshRenderer, MeshFilter)
│   ├── CollisionMesh (MeshCollider)
│   └── JointController (Script)
├── Link1/
│   ├── VisualMesh
│   ├── CollisionMesh
│   └── JointController
└── Link2/
    ├── VisualMesh
    ├── CollisionMesh
    └── JointController
```

### Joint Controller Script

```csharp
using UnityEngine;

[RequireComponent(typeof(ConfigurableJoint))]
public class JointController : MonoBehaviour
{
    [Header("Joint Configuration")]
    public string jointName;
    public JointType jointType = JointType.Revolute;
    public float jointPosition = 0f;
    public float jointVelocity = 0f;

    [Header("Limits")]
    public float lowerLimit = -Mathf.PI/2;
    public float upperLimit = Mathf.PI/2;
    public float maxEffort = 100f;

    private ConfigurableJoint joint;
    private Rigidbody rb;

    void Start()
    {
        joint = GetComponent<ConfigurableJoint>();
        rb = GetComponent<Rigidbody>();
        ConfigureJoint();
    }

    void ConfigureJoint()
    {
        switch (jointType)
        {
            case JointType.Revolute:
                ConfigureRevoluteJoint();
                break;
            case JointType.Prismatic:
                ConfigurePrismaticJoint();
                break;
            case JointType.Fixed:
                ConfigureFixedJoint();
                break;
        }
    }

    void ConfigureRevoluteJoint()
    {
        joint.xMotion = ConfigurableJointMotion.Locked;
        joint.yMotion = ConfigurableJointMotion.Locked;
        joint.zMotion = ConfigurableJointMotion.Locked;
        joint.angularXMotion = ConfigurableJointMotion.Limited;
        joint.angularYMotion = ConfigurableJointMotion.Locked;
        joint.angularZMotion = ConfigurableJointMotion.Locked;

        SoftJointLimit limit = new SoftJointLimit();
        limit.limit = upperLimit;
        joint.highAngularXLimit = limit;

        limit.limit = -lowerLimit;
        joint.lowAngularXLimit = limit;
    }

    public void SetJointPosition(float position)
    {
        // Clamp position to limits
        jointPosition = Mathf.Clamp(position, lowerLimit, upperLimit);

        // Apply rotation based on joint position
        transform.localEulerAngles = new Vector3(
            jointPosition * Mathf.Rad2Deg, 0, 0
        );
    }

    public float GetJointPosition()
    {
        return jointPosition;
    }
}

public enum JointType
{
    Fixed,
    Revolute,
    Prismatic,
    Continuous,
    Planar
}
```

## Unity-Gazebo Integration

### Using Unity as Visualization Layer

Unity can serve as a visualization layer while Gazebo handles physics:

```csharp
using UnityEngine;
using ROS2;
using nav_msgs;
using geometry_msgs;

public class GazeboUnityBridge : MonoBehaviour
{
    private ROS2UnityComponent ros2Unity;
    private GameObject[] robotLinks;

    void Start()
    {
        ros2Unity = GetComponent<ROS2UnityComponent>();
        ros2Unity.Initialize();

        // Subscribe to Gazebo model states
        ros2Unity.CreateSubscription<ModelStates>(
            "/gazebo/model_states", ModelStatesCallback
        );

        // Get robot link objects
        robotLinks = GameObject.FindGameObjectsWithTag("RobotLink");
    }

    void ModelStatesCallback(ModelStates modelStates)
    {
        for (int i = 0; i < modelStates.name.Count; i++)
        {
            string modelName = modelStates.name[i];
            var pose = modelStates.pose[i];

            // Find corresponding Unity object
            GameObject linkObject = GameObject.Find(modelName);
            if (linkObject != null)
            {
                // Update position and orientation
                linkObject.transform.position = new Vector3(
                    (float)pose.position.x,
                    (float)pose.position.z,  // Unity Y is up, ROS Z is up
                    -(float)pose.position.y  // Unity Z is forward, ROS Y is forward
                );

                linkObject.transform.rotation = new Quaternion(
                    (float)pose.orientation.x,
                    (float)pose.orientation.z,
                    -(float)pose.orientation.y,  // Unity Z is forward, ROS Y is forward
                    (float)pose.orientation.w
                );
            }
        }
    }
}
```

## Advanced Visualization Techniques

### Custom Shaders for Robot Materials

```hlsl
// RobotMaterial.shader
Shader "Robot/RobotMaterial"
{
    Properties
    {
        _Color ("Color", Color) = (0.5, 0.5, 0.5, 1)
        _Metallic ("Metallic", Range(0, 1)) = 0.5
        _Smoothness ("Smoothness", Range(0, 1)) = 0.5
        _EmissionColor ("Emission Color", Color) = (0, 0, 0, 1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 200

        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows
        #pragma target 3.0

        struct Input
        {
            float2 uv_MainTex;
        };

        fixed4 _Color;
        half _Metallic;
        half _Smoothness;
        fixed4 _EmissionColor;

        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            o.Albedo = _Color.rgb;
            o.Metallic = _Metallic;
            o.Smoothness = _Smoothness;
            o.Emission = _EmissionColor.rgb;
            o.Alpha = _Color.a;
        }
        ENDCG
    }
    FallBack "Diffuse"
}
```

### Animation and State Visualization

```csharp
using UnityEngine;

public class RobotStateVisualizer : MonoBehaviour
{
    [Header("State Indicators")]
    public Light[] statusLights;
    public Material[] statusMaterials;
    public Animation[] stateAnimations;

    [Header("Sensor Visualization")]
    public GameObject[] sensorBeams;
    public GameObject[] detectionVolumes;

    private RobotState currentState = RobotState.Idle;

    public enum RobotState
    {
        Idle,
        Moving,
        Sensing,
        Processing,
        Error
    }

    void Update()
    {
        UpdateStateVisualization();
    }

    void UpdateStateVisualization()
    {
        switch (currentState)
        {
            case RobotState.Idle:
                SetStatusLightColor(Color.blue);
                SetMaterialColor(Color.gray);
                break;
            case RobotState.Moving:
                SetStatusLightColor(Color.green);
                SetMaterialColor(Color.cyan);
                break;
            case RobotState.Sensing:
                SetStatusLightColor(Color.yellow);
                SetMaterialColor(Color.yellow);
                ActivateSensorBeams();
                break;
            case RobotState.Error:
                SetStatusLightColor(Color.red);
                SetMaterialColor(Color.red);
                break;
        }
    }

    void SetStatusLightColor(Color color)
    {
        foreach (var light in statusLights)
        {
            light.color = color;
        }
    }

    void SetMaterialColor(Color color)
    {
        foreach (var renderer in GetComponentsInChildren<Renderer>())
        {
            renderer.material.color = color;
        }
    }

    void ActivateSensorBeams()
    {
        foreach (var beam in sensorBeams)
        {
            beam.SetActive(true);
        }
    }
}
```

## Best Practices

### Performance Optimization

1. **LOD System**: Use Level of Detail for complex robots
2. **Occlusion Culling**: Implement occlusion culling for large environments
3. **Object Pooling**: Reuse objects for dynamic elements
4. **Shader Optimization**: Use efficient shaders for real-time rendering
5. **Batching**: Use static and dynamic batching where appropriate

### Data Synchronization

1. **Update Rates**: Match Unity update rates with ROS message rates
2. **Interpolation**: Smooth transitions between received states
3. **Buffering**: Handle message delays and out-of-order delivery
4. **Threading**: Use appropriate threading for ROS communication
5. **Error Handling**: Gracefully handle connection issues

### Visualization Design

1. **Consistency**: Maintain consistent coordinate systems
2. **Scale**: Use appropriate scales for visualization
3. **Color Coding**: Use meaningful color schemes for different elements
4. **Transparency**: Use transparency for overlapping elements
5. **Labels**: Include clear labels for complex visualizations

## Integration Strategies

### Real-time vs. Offline Visualization

- **Real-time**: Direct ROS communication, immediate updates
- **Offline**: Process recorded data, playback at desired speed
- **Hybrid**: Combine both approaches for development and analysis

### Multi-robot Visualization

```csharp
using System.Collections.Generic;

public class MultiRobotVisualizer : MonoBehaviour
{
    private Dictionary<string, GameObject> robotObjects = new Dictionary<string, GameObject>();

    public void AddRobot(string robotName, GameObject robotPrefab)
    {
        if (!robotObjects.ContainsKey(robotName))
        {
            GameObject robotInstance = Instantiate(robotPrefab);
            robotObjects[robotName] = robotInstance;
            robotInstance.name = robotName;
        }
    }

    public void UpdateRobotPose(string robotName, Vector3 position, Quaternion rotation)
    {
        if (robotObjects.ContainsKey(robotName))
        {
            robotObjects[robotName].transform.position = position;
            robotObjects[robotName].transform.rotation = rotation;
        }
    }
}
```

Unity provides powerful visualization capabilities that complement physics-based simulators like Gazebo, enabling rich, interactive robot development and debugging experiences.
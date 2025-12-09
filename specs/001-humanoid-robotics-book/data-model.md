# Data Model: Physical AI & Humanoid Robotics Book

## Key Entities from Feature Specification

### Humanoid Robot
- **Description**: A physical or simulated robot with human-like form, controlled by various software components.
- **Attributes (Conceptual)**:
    - `id`: Unique identifier for the robot.
    - `model_type`: Type of humanoid model (e.g., specific URDF/SDF)._
    - `state`: Current operational state (e.g., idle, walking, grasping).
    - `sensors`: Collection of attached sensors (LiDAR, Depth Cameras, IMUs).
    - `actuators`: Collection of controlled actuators (motors, servos).
    - `pose`: Positional and orientational data.
- **Relationships**: Interacts with `ROS 2 Node`, `Simulation Environment`, `Sensors`, `Actuators`, `LLM/VLA`.

### ROS 2 Node
- **Description**: An executable process within the ROS 2 graph responsible for specific robot functionalities.
- **Attributes (Conceptual)**:
    - `node_name`: Unique name within the ROS 2 graph.
    - `node_type`: Functionality (e.g., publisher, subscriber, service server/client).
    - `language`: Implementation language (e.g., Python).
- **Relationships**: Communicates via `Topics`, `Services`, `Actions`.

### Simulation Environment
- **Description**: Digital platforms (Gazebo, Unity, Isaac Sim) used to model and test robot behavior in virtual settings.
- **Attributes (Conceptual)**:
    - `environment_type`: (e.g., Gazebo, Unity, Isaac Sim).
    - `physical_properties`: Gravity, collision parameters.
    - `assets`: 3D models of robots and environments.
- **Relationships**: Hosts `Humanoid Robot`, simulates `Sensors`, integrates with `ROS 2 Node`.

### Sensors
- **Description**: Devices (LiDAR, Depth Cameras, IMUs) that gather data from the environment, either physical or simulated.
- **Attributes (Conceptual)**:
    - `sensor_type`: (e.g., LiDAR, Depth Camera, IMU).
    - `data_format`: Output data structure.
    - `calibration`: Calibration parameters.
- **Relationships**: Provides input to `Humanoid Robot` (via ROS 2 Nodes), simulated within `Simulation Environment`.

### Actuators
- **Description**: Components (e.g., motors, servos) that enable physical movement of the robot.
- **Attributes (Conceptual)**:
    - `actuator_type`: (e.g., motor, servo).
    - `control_interface`: Method of control (e.g., ROS 2 topic, service).
- **Relationships**: Receives commands from `Humanoid Robot` (via ROS 2 Nodes).

### LLM/VLA
- **Description**: Large Language Models and Vision-Language-Action models used for cognitive robotics and natural language interaction.
- **Attributes (Conceptual)**:
    - `model_name`: Specific LLM/VLA model (e.g., OpenAI Whisper).
    - `capabilities`: Speech recognition, natural language processing, planning.
- **Relationships**: Provides cognitive input to `Humanoid Robot`, translates natural language to `ROS 2 Actions`.

### Docusaurus
- **Description**: A static site generator used to publish the book content.
- **Attributes (Conceptual)**:
    - `configuration`: Site structure, navigation, themes.
    - `content_format`: Markdown.
- **Relationships**: Publishes `Book Content` to `GitHub Pages`.

### GitHub Pages
- **Description**: A hosting service for deploying web content directly from a GitHub repository.
- **Attributes (Conceptual)**:
    - `repository_link`: URL to the GitHub repository.
    - `deployment_status`: Current deployment state.
- **Relationships**: Hosts `Docusaurus` generated `Book Content`.


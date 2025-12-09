# ROS 2 Architecture

## Overview

ROS 2 (Robot Operating System 2) is designed to be a flexible framework for writing robot software. It's not an operating system but rather a middleware that provides services designed for a heterogeneous computer cluster, including hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more.

## Key Concepts

### Nodes
A node is a process that performs computation. ROS 2 is designed with the philosophy that a single process should only perform a single task. This means that a robot application is typically composed of many nodes, each performing a specific task like sensor processing, motor control, or localization.

### Packages
Packages are the software organization unit in ROS 2. Each package can contain libraries, executables, configuration files, and other resources needed for a specific task. Packages can depend on other packages and can be easily shared and reused.

### Topics and Messages
Topics are named buses over which nodes exchange messages. Messages are the data format exchanged between nodes. ROS 2 uses a publish-subscribe communication pattern where nodes publish messages to topics and other nodes subscribe to topics to receive those messages.

### Services
Services provide a request-response communication pattern. A node can offer a service that other nodes can call to request specific functionality. This is useful for operations that need to be completed before continuing.

### Actions
Actions are similar to services but are designed for long-running tasks. They provide feedback during execution and can be canceled if needed. This is particularly useful for navigation tasks or complex manipulations.

## Communication Patterns

### Publish-Subscribe (Topics)
- Used for streaming data
- One-to-many communication
- Asynchronous
- Examples: sensor data, robot state

### Request-Response (Services)
- Used for simple synchronous operations
- One-to-one communication
- Request followed by response
- Examples: saving a map, changing parameters

### Action Interface
- Used for long-running tasks
- One-to-one communication with feedback
- Can be preempted
- Examples: navigation, trajectory execution

## ROS 2 Middleware

ROS 2 uses DDS (Data Distribution Service) as its underlying communication middleware. DDS provides the publish-subscribe communication pattern and handles message delivery between nodes, even across different machines.

## Quality of Service (QoS)

ROS 2 provides Quality of Service settings that allow you to tune communication behavior based on your application's needs:
- Reliability: Best effort vs. reliable delivery
- Durability: Volatile vs. transient local
- History: Keep all messages vs. keep last N messages
- Deadline: Maximum time between messages
- Lifespan: Maximum time a message is valid
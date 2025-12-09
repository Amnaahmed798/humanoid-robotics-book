# Python rclpy Integration

## Introduction to rclpy

`rclpy` is the Python client library for ROS 2. It provides the Python API that allows you to create ROS 2 nodes, publish and subscribe to topics, provide and use services, and work with actions. This library is essential for developing ROS 2 applications in Python.

## Installing rclpy

`rclpy` is typically installed as part of the ROS 2 distribution. If you're using a standard ROS 2 installation, it should already be available. You can verify the installation with:

```bash
python3 -c "import rclpy; print(rclpy.__version__)"
```

## Creating a Simple Node

Here's the basic structure of a ROS 2 node using rclpy:

```python
import rclpy
from rclpy.node import Node

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node_name')
        # Node initialization code here
```

## Publishers

To create a publisher that sends messages to a topic:

```python
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher = self.create_publisher(String, 'topic_name', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1
```

## Subscribers

To create a subscriber that receives messages from a topic:

```python
from std_msgs.msg import String

class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'topic_name',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')
```

## Services

To create a service server:

```python
from example_interfaces.srv import AddTwoInts

class ServiceServer(Node):
    def __init__(self):
        super().__init__('service_server')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {response.sum}')
        return response
```

To create a service client:

```python
from example_interfaces.srv import AddTwoInts

class ServiceClient(Node):
    def __init__(self):
        super().__init__('service_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Parameters

ROS 2 allows you to configure nodes using parameters:

```python
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('param_name', 'default_value')
        self.declare_parameter('number_param', 42)

        # Get parameter values
        param_value = self.get_parameter('param_name').value
        number_value = self.get_parameter('number_param').value
```

## Best Practices

1. Always use proper error handling and clean shutdown procedures
2. Use appropriate Quality of Service (QoS) settings for your use case
3. Follow ROS 2 naming conventions (snake_case for nodes, topics, services)
4. Use standard message types when possible
5. Include proper logging for debugging
6. Use parameters for configuration rather than hardcoding values
#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class Create3Teleop(Node):
    def __init__(self):
        super().__init__("create3_teleop")
        # Publish to Create3 cmd_vel
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.get_logger().info("Create3Teleop node ready. Publishing to /cmd_vel.")

    def send_twist(self, linear_x: float, angular_z: float, duration: float):
        """
        Send a Twist command for `duration` seconds, then stop.
        """
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)

        stop = Twist()

        self.get_logger().info(
            f"Sending cmd_vel: linear.x={twist.linear.x:.3f}, angular.z={twist.angular.z:.3f} "
            f"for {duration:.2f} seconds"
        )

        end_time = time.time() + duration
        rate_hz = 20.0
        period = 1.0 / rate_hz

        while time.time() < end_time:
            self.pub.publish(twist)
            time.sleep(period)

        # Send zero command to stop
        self.pub.publish(stop)
        self.get_logger().info("Stopped.")


def main():
    rclpy.init()

    node = Create3Teleop()

    print("=== Create3 CLI Teleop ===")
    print("Robot must be on the floor, off the dock, ROS_DOMAIN_ID must match the Create3.")
    print()
    print("Commands (press Enter after each):")
    print("  f   -> move forward 0.3 m")
    print("  b   -> move backward 0.3 m")
    print("  l   -> turn left 45 degrees")
    print("  r   -> turn right 45 degrees")
    print("  s   -> stop (send zero Twist once)")
    print("  q   -> quit")
    print()

    try:
        while True:
            try:
                cmd = input("teleop> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if cmd in ("q", "quit", "exit"):
                print("Goodbye.")
                break

            if cmd == "f":
                # forward: small distance
                node.send_twist(linear_x=0.1, angular_z=0.0, duration=3.0)  # ~0.3m at 0.1 m/s
            elif cmd == "b":
                node.send_twist(linear_x=-0.1, angular_z=0.0, duration=3.0)
            elif cmd == "l":
                # Turn left in place (tune angular speed/duration to get ~45 deg)
                node.send_twist(linear_x=0.0, angular_z=0.25 , duration=2.0)
            elif cmd == "r":
                node.send_twist(linear_x=0.0, angular_z=-0.25, duration=2.0)
            elif cmd == "s":
                # One zero Twist to be safe
                node.send_twist(linear_x=0.0, angular_z=0.0, duration=0.1)
            elif cmd == "":
                continue
            else:
                print("Unknown command. Use f/b/l/r/s/q.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

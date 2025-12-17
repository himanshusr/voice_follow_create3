#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from robot.vision import describe_scene_with_gemini

from robot.intent import parse_intent


# Tunable motion constants
MOVE_SPEED_M_S = 0.1     # linear speed (m/s)
TURN_SPEED_RAD_S = 0.25  # angular speed (rad/s)
MIN_DURATION = 0.2       # safety floor so we actually see motion


class Create3NLTeleop(Node):
    def __init__(self):
        super().__init__("create3_nl_teleop")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.get_logger().info("Create3NLTeleop node ready. Publishing to /cmd_vel.")

    def _send_twist(self, linear_x: float, angular_z: float, duration: float):
        """
        Send a Twist for `duration` seconds, then stop.
        """
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)

        stop = Twist()

        duration = max(MIN_DURATION, float(duration))
        self.get_logger().info(
            f"cmd_vel: lin={twist.linear.x:.3f} m/s, ang={twist.angular.z:.3f} rad/s, "
            f"for {duration:.2f} s"
        )

        end_time = time.time() + duration
        rate_hz = 20.0
        period = 1.0 / rate_hz

        while time.time() < end_time:
            self.pub.publish(twist)
            time.sleep(period)

        self.pub.publish(stop)
        self.get_logger().info("Stopped this step.")

    def execute_movement_sequence(self, actions):
        """
        Execute a parsed movement sequence from parse_intent().
        """
        for i, action in enumerate(actions, start=1):
            atype = action.get("type")
            self.get_logger().info(f"Step {i}: {action}")

            if atype == "move":
                direction = action.get("direction", "forward")
                dist_m = float(action.get("distance_m", 0.3))

                # compute duration from distance and speed
                duration = abs(dist_m) / MOVE_SPEED_M_S
                lin = MOVE_SPEED_M_S if direction == "forward" else -MOVE_SPEED_M_S
                self._send_twist(linear_x=lin, angular_z=0.0, duration=duration)

            elif atype == "turn":
                direction = action.get("direction", "right")
                degrees = float(action.get("degrees", 90.0))

                # convert degrees to radians and compute duration
                angle_rad = math.radians(abs(degrees))
                duration = angle_rad / TURN_SPEED_RAD_S
                ang = TURN_SPEED_RAD_S if direction == "left" else -TURN_SPEED_RAD_S
                self._send_twist(linear_x=0.0, angular_z=ang, duration=duration)

            elif atype == "stop":
                # Send one small zero twist
                self._send_twist(linear_x=0.0, angular_z=0.0, duration=0.1)
            else:
                self.get_logger().warn(f"Unknown action type: {atype}")


def format_actions_for_speech(actions):
    """
    Just for printing a friendly summary in the CLI.
    """
    parts = []
    for action in actions:
        atype = action.get("type")

        if atype == "move":
            direction = action.get("direction", "forward")
            dist = action.get("distance_m")
            if dist is not None:
                try:
                    f = float(dist)
                    dist_str = str(int(f)) if f.is_integer() else f"{f:g}"
                except Exception:
                    dist_str = str(dist)
                parts.append(f"move {direction} {dist_str} meters")
            else:
                parts.append(f"move {direction}")

        elif atype == "turn":
            direction = action.get("direction", "right")
            deg = action.get("degrees")
            if deg is not None:
                try:
                    f = float(deg)
                    deg_str = str(int(f)) if f.is_integer() else f"{f:g}"
                except Exception:
                    deg_str = str(deg)
                parts.append(f"turn {direction} {deg_str} degrees")
            else:
                parts.append(f"turn {direction}")

        elif atype == "stop":
            parts.append("stop")

    if not parts:
        return "executing your movement sequence"

    if len(parts) == 1:
        phrase = parts[0]
        if phrase.startswith("move "):
            phrase = "moving " + phrase[len("move "):]
        elif phrase.startswith("turn "):
            phrase = "turning " + phrase[len("turn "):]
        elif phrase == "stop":
            phrase = "stopping"
        return phrase

    if len(parts) == 2:
        first, second = parts
        if first.startswith("move "):
            first = "moving " + first[len("move "):]
        elif first.startswith("turn "):
            first = "turning " + first[len("turn "):]
        return f"{first} and then {second}"

    return ", then ".join(parts)


def main():
    rclpy.init()
    node = Create3NLTeleop()

    print("=== Natural Language Create3 Teleop ===")
    print("Examples:")
    print("  move forward 0.5 meters")
    print("  move forward 1 meter and turn right 90 degrees")
    print("  go back 0.3 meters")
    print("  turn left 45 degrees")
    print("  stop")
    print("Type 'quit' or 'q' to exit.")
    print()

    try:
        while True:
            try:
                text = input("nl> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if text.lower() in ("q", "quit", "exit"):
                print("Goodbye.")
                break

            if not text:
                continue

            intent = parse_intent(text)
            print(f"[DEBUG] intent: {intent}")

            kind = intent.get("intent", "unknown")

            if kind == "movement_sequence":
                actions = intent.get("actions", [])
                if not actions:
                    print("[WARN] No actions found in movement_sequence.")
                    continue

                summary = format_actions_for_speech(actions)
                print(f"[ROBOT] {summary}...")
                node.execute_movement_sequence(actions)

            elif kind == "stop":
                print("[ROBOT] Stopping.")
                node.execute_movement_sequence([{"type": "stop"}])

            elif kind == "describe_surroundings":
                print("[VISION] Capturing image and asking Gemini...")
                try:
                    description = describe_scene_with_gemini(device_index=0)
                    print("\n[VISION RESULT]")
                    print(description)
                    print()
                except Exception as e:
                    print(f"[VISION ERROR] {e}")

            else:
                print("[WARN] Unknown intent, nothing to do.")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

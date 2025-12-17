#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from robot.intent import parse_intent
from robot.speech import WhisperRecognizer
from robot.vision import describe_scene_with_gemini

# --- same motion tuning as nl_teleop.py ---

MOVE_SPEED_M_S = 0.1
TURN_SPEED_RAD_S = 0.25
MIN_DURATION = 0.2


class Create3VoiceTeleop(Node):
    def __init__(self):
        super().__init__("create3_voice_teleop")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.get_logger().info("Create3VoiceTeleop node ready. Publishing to /cmd_vel.")

    def _send_twist(self, linear_x: float, angular_z: float, duration: float):
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
        for i, action in enumerate(actions, start=1):
            atype = action.get("type")
            self.get_logger().info(f"Step {i}: {action}")

            if atype == "move":
                direction = action.get("direction", "forward")
                dist_m = float(action.get("distance_m", 0.3))

                duration = abs(dist_m) / MOVE_SPEED_M_S
                lin = MOVE_SPEED_M_S if direction == "forward" else -MOVE_SPEED_M_S
                self._send_twist(linear_x=lin, angular_z=0.0, duration=duration)

            elif atype == "turn":
                direction = action.get("direction", "right")
                degrees = float(action.get("degrees", 90.0))

                angle_rad = math.radians(abs(degrees))
                duration = angle_rad / TURN_SPEED_RAD_S
           

                ang = TURN_SPEED_RAD_S if direction == "left" else -TURN_SPEED_RAD_S
                self._send_twist(linear_x=0.0, angular_z=ang, duration=duration)

            elif atype == "stop":
                self._send_twist(linear_x=0.0, angular_z=0.0, duration=0.1)
            else:
                self.get_logger().warn(f"Unknown action type: {atype}")


def format_actions_for_speech(actions):
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
    node = Create3VoiceTeleop()

    recognizer = WhisperRecognizer(
        model_name="tiny",
        sample_rate=16000,
        arecord_device="plughw:0,0",  # matches your working USB mic
    )

    print("\n=== Voice Create3 Teleop ===")
    print("Flow:")
    print("  1) Press Enter → robot listens.")
    print("  2) Say a command like:")
    print('       "move forward 0.5 meters"')
    print('       "move forward 0.3 meters and turn right 90 degrees"')
    print('       "turn left 45 degrees"')
    print('       "stop"')
    print("  3) Wait for transcription + motion.")
    print("Type 'q' on the prompt to quit.\n")

    try:
        while True:
            try:
                cmd = input("Press Enter to speak, or type 'q' to quit: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if cmd in ("q", "quit", "exit"):
                print("Goodbye.")
                break

            # Trigger mic capture
            text = recognizer.listen_and_transcribe(
                max_seconds=8.0,
                silence_threshold=500.0,
                silence_duration=1.0,
            )

            if not text:
                print("[VOICE] I didn't hear anything.")
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

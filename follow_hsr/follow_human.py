#!/usr/bin/env python3
"""
Human Following Robot for iRobot Create3

This script enables the Create3 robot to detect and follow a human using
computer vision. It uses MediaPipe for person detection and publishes
velocity commands to /cmd_vel.

Usage:
    source /opt/ros/jazzy/setup.bash
    cd ~/robot_ai
    source .venv/bin/activate
    python follow_hsr/follow_human.py

Controls:
    - 'q' or Ctrl+C: Quit
    - 's': Toggle search mode when person lost
    - 'd': Toggle debug visualization window
    - 'p': Pause/resume following
"""

import os
import sys
import time
import argparse
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import PersonDetector, PersonDetection
from controller import FollowController, SearchController, ControllerConfig

# Debug output directory
DEBUG_DIR = Path(__file__).parent / "debug_output"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


class HumanFollower(Node):
    """
    ROS 2 node that follows a human using camera vision.
    """

    def __init__(
        self,
        camera_index: int = 0,
        show_video: bool = True,
        use_mediapipe: bool = True,
        config: ControllerConfig = None,
        save_images: bool = False,
        save_interval: float = 2.0,
    ):
        super().__init__("human_follower")

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Camera
        self.camera_index = camera_index
        self.cap = None
        self._init_camera()

        # Detection & Control
        self.detector = PersonDetector(use_mediapipe=use_mediapipe)
        self.controller = FollowController(config or ControllerConfig())
        self.search_controller = SearchController()

        # State
        self.show_video = show_video
        self.is_paused = False
        self.search_mode = True  # Search when person lost
        self.running = True
        self.last_detection: Optional[PersonDetection] = None
        self.frames_without_detection = 0

        # Timing
        self.last_time = time.time()
        self.loop_rate = 15.0  # Hz

        # Debug image saving
        self.save_images = save_images
        self.save_interval = save_interval
        self.last_save_time = 0.0
        self.frame_count = 0

        self.get_logger().info("=" * 60)
        self.get_logger().info("HumanFollower node initialized")
        self.get_logger().info(f"  Camera index: {camera_index}")
        self.get_logger().info(f"  Using MediaPipe: {self.detector.use_mediapipe}")
        self.get_logger().info(f"  Show video: {show_video}")
        self.get_logger().info(f"  Save debug images: {save_images}")
        self.get_logger().info("=" * 60)

    def _init_camera(self):
        """Initialize the camera."""
        self.get_logger().info(f"Opening camera {self.camera_index}...")
        
        # Try V4L2 first on Linux
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {self.camera_index}")

        # Set resolution (lower = faster)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Warm up camera
        for _ in range(10):
            self.cap.read()

        self.get_logger().info("Camera ready.")

    def _send_velocity(self, linear: float, angular: float):
        """Publish velocity command to /cmd_vel."""
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        self.cmd_pub.publish(twist)

    def _stop(self):
        """Stop the robot."""
        self._send_velocity(0.0, 0.0)

    def _process_frame(self, frame: np.ndarray) -> Optional[PersonDetection]:
        """Process a camera frame and return detection."""
        return self.detector.detect(frame)

    def _handle_keyboard(self, key: int) -> bool:
        """
        Handle keyboard input.
        Returns False if should quit, True otherwise.
        """
        if key == ord('q') or key == 27:  # 'q' or ESC
            return False
        elif key == ord('s'):
            self.search_mode = not self.search_mode
            mode = "ON" if self.search_mode else "OFF"
            self.get_logger().info(f"Search mode: {mode}")
        elif key == ord('d'):
            self.show_video = not self.show_video
            if not self.show_video:
                cv2.destroyAllWindows()
            self.get_logger().info(f"Video display: {'ON' if self.show_video else 'OFF'}")
        elif key == ord('p'):
            self.is_paused = not self.is_paused
            if self.is_paused:
                self._stop()
            self.get_logger().info(f"Following: {'PAUSED' if self.is_paused else 'ACTIVE'}")
        elif key == ord(' '):  # Space = emergency stop
            self._stop()
            self.is_paused = True
            self.get_logger().warn("EMERGENCY STOP - Press 'p' to resume")

        return True

    def _save_debug_frame(self, frame: np.ndarray, detection: Optional[PersonDetection], 
                          linear_vel: float, angular_vel: float, status: str):
        """Save annotated debug frame."""
        display = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw center crosshair
        cx, cy = w // 2, h // 2
        cv2.line(display, (cx, 0), (cx, h), (100, 100, 100), 1)
        
        if detection:
            # Draw bounding box
            x, y, bw, bh = detection.bbox
            cv2.rectangle(display, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(display, (detection.center_x, detection.center_y), 8, (0, 0, 255), -1)
            cv2.line(display, (cx, cy), (detection.center_x, detection.center_y), (255, 255, 0), 2)
        
        # Status and velocity
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display, f"lin: {linear_vel:+.2f} ang: {angular_vel:+.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        timestamp = datetime.now().strftime("%H%M%S")
        path = DEBUG_DIR / f"follow_{timestamp}.jpg"
        cv2.imwrite(str(path), display)
        return path

    def spin_once(self) -> bool:
        """
        Run one iteration of the control loop.
        Returns False if should stop, True otherwise.
        """
        # Calculate dt
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        self.frame_count += 1

        # Read camera frame
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn("Failed to read camera frame")
            return True

        # Detect person
        detection = self._process_frame(frame)
        self.last_detection = detection

        # Track frames without detection
        if detection is None:
            self.frames_without_detection += 1
        else:
            self.frames_without_detection = 0

        # Compute velocity command
        if self.is_paused:
            linear_vel, angular_vel = 0.0, 0.0
            status = "PAUSED"
        elif detection is not None:
            linear_vel, angular_vel = self.controller.compute(detection, dt)
            status = "FOLLOWING"
        elif self.search_mode and self.frames_without_detection > 10:
            # Search for person
            linear_vel, angular_vel = self.search_controller.compute(dt)
            status = "SEARCHING"
        else:
            linear_vel, angular_vel = 0.0, 0.0
            status = "NO PERSON"

        # Send velocity command
        self._send_velocity(linear_vel, angular_vel)

        # === DETAILED LOGGING (every 15 frames ~1 second) ===
        if self.frame_count % 15 == 0:
            if detection:
                # Person detected - show details
                direction = "LEFT" if detection.normalized_x < -0.1 else "RIGHT" if detection.normalized_x > 0.1 else "CENTER"
                distance = "CLOSE" if detection.normalized_area > 0.15 else "FAR" if detection.normalized_area < 0.05 else "OK"
                self.get_logger().info(
                    f"[{status}] Person: {direction} (x={detection.normalized_x:+.2f}) | "
                    f"Dist: {distance} (area={detection.normalized_area:.3f}) | "
                    f"Cmd: lin={linear_vel:+.2f} ang={angular_vel:+.2f}"
                )
            else:
                self.get_logger().info(
                    f"[{status}] No person detected (lost for {self.frames_without_detection} frames) | "
                    f"Cmd: lin={linear_vel:+.2f} ang={angular_vel:+.2f}"
                )

        # === SAVE DEBUG IMAGES periodically ===
        if self.save_images and (current_time - self.last_save_time) > self.save_interval:
            path = self._save_debug_frame(frame, detection, linear_vel, angular_vel, status)
            self.get_logger().info(f"[DEBUG] Saved: {path}")
            self.last_save_time = current_time

        # Visualization (if display available)
        if self.show_video:
            display_frame = frame.copy()

            if detection:
                display_frame = self.detector.draw_detection(display_frame, detection)
                color = (0, 255, 0)
            elif self.is_paused:
                color = (0, 255, 255)
            elif self.search_mode and self.frames_without_detection > 10:
                color = (255, 165, 0)
            else:
                color = (0, 0, 255)

            # Draw status
            cv2.putText(display_frame, status, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Draw velocity info
            vel_text = f"lin: {linear_vel:.2f} m/s, ang: {angular_vel:.2f} rad/s"
            cv2.putText(display_frame, vel_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Human Follower", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if not self._handle_keyboard(key):
                return False

        return True

    def run(self):
        """Main control loop."""
        self.get_logger().info("=" * 50)
        self.get_logger().info("Human Following Robot Started!")
        self.get_logger().info("=" * 50)
        self.get_logger().info("Controls:")
        self.get_logger().info("  'q' or ESC  - Quit")
        self.get_logger().info("  'p'         - Pause/Resume")
        self.get_logger().info("  's'         - Toggle search mode")
        self.get_logger().info("  'd'         - Toggle video display")
        self.get_logger().info("  SPACE       - Emergency stop")
        self.get_logger().info("=" * 50)

        rate_period = 1.0 / self.loop_rate

        try:
            while self.running and rclpy.ok():
                loop_start = time.time()

                # Process ROS callbacks
                rclpy.spin_once(self, timeout_sec=0.001)

                # Run control loop
                if not self.spin_once():
                    break

                # Rate limiting
                elapsed = time.time() - loop_start
                sleep_time = rate_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            self.get_logger().info("Interrupted by user")
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown."""
        self.get_logger().info("Shutting down...")
        self.running = False
        self._stop()

        if self.cap:
            self.cap.release()
        
        self.detector.close()
        cv2.destroyAllWindows()
        
        self.get_logger().info("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Human Following Robot for Create3")
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video display window"
    )
    parser.add_argument(
        "--no-mediapipe",
        action="store_true",
        help="Use OpenCV HOG instead of MediaPipe"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save debug images periodically (for headless debugging)"
    )
    parser.add_argument(
        "--save-interval",
        type=float,
        default=2.0,
        help="Seconds between debug image saves (default: 2.0)"
    )
    parser.add_argument(
        "--target-distance",
        type=float,
        default=0.08,
        help="Target bbox area (proxy for distance, default: 0.08)"
    )
    parser.add_argument(
        "--max-linear-speed",
        type=float,
        default=0.2,
        help="Maximum linear speed in m/s (default: 0.2)"
    )
    parser.add_argument(
        "--max-angular-speed",
        type=float,
        default=1.0,
        help="Maximum angular speed in rad/s (default: 1.0)"
    )

    args = parser.parse_args()

    # Build config from args
    config = ControllerConfig(
        target_area=args.target_distance,
        max_linear_speed=args.max_linear_speed,
        max_angular_speed=args.max_angular_speed,
    )

    # Initialize ROS 2
    rclpy.init()

    # Handle SIGINT gracefully
    follower = None

    def signal_handler(sig, frame):
        if follower:
            follower.running = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        follower = HumanFollower(
            camera_index=args.camera,
            show_video=not args.no_video,
            use_mediapipe=not args.no_mediapipe,
            config=config,
            save_images=args.save_images,
            save_interval=args.save_interval,
        )
        follower.run()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        if follower:
            follower.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


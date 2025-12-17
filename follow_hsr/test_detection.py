#!/usr/bin/env python3
"""
Test script for debugging person detection - NO ROS REQUIRED!

Run this to:
  1. Capture a photo from your camera
  2. Run person detection
  3. See the annotated result
  4. Save debug images

Usage:
    python follow_hsr/test_detection.py              # Live camera test
    python follow_hsr/test_detection.py --snapshot   # Take one photo and analyze
    python follow_hsr/test_detection.py --image path/to/image.jpg  # Test on existing image
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# Add current directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import PersonDetector, PersonDetection, MEDIAPIPE_AVAILABLE

# Output directory for debug images
DEBUG_DIR = Path("follow_hsr/debug_output")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str, level: str = "INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {msg}")


def test_camera(camera_index: int = 0) -> bool:
    """Test if camera can be opened."""
    log(f"Testing camera at index {camera_index}...")
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        log(f"FAILED to open camera {camera_index}", "ERROR")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame is not None:
        h, w = frame.shape[:2]
        log(f"Camera OK! Resolution: {w}x{h}")
        return True
    else:
        log("Camera opened but failed to read frame", "ERROR")
        return False


def capture_frame(camera_index: int = 0, warmup: int = 5) -> np.ndarray:
    """Capture a single frame from camera."""
    log(f"Opening camera {camera_index}...")
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    log(f"Warming up camera ({warmup} frames)...")
    for i in range(warmup):
        cap.read()
    
    log("Capturing frame...")
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame")
    
    h, w = frame.shape[:2]
    log(f"Captured frame: {w}x{h}")
    return frame


def save_debug_image(frame: np.ndarray, suffix: str = "") -> Path:
    """Save frame with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_{timestamp}{suffix}.jpg"
    path = DEBUG_DIR / filename
    cv2.imwrite(str(path), frame)
    log(f"Saved: {path}")
    return path


def analyze_detection(detection: PersonDetection | None, frame_shape: tuple):
    """Print detailed detection analysis."""
    h, w = frame_shape[:2]
    
    print("\n" + "=" * 60)
    print("DETECTION ANALYSIS")
    print("=" * 60)
    
    if detection is None:
        print("❌ NO PERSON DETECTED")
        print("\nPossible reasons:")
        print("  - No person in frame")
        print("  - Person too far away")
        print("  - Poor lighting")
        print("  - Person partially occluded")
        return
    
    print("✅ PERSON DETECTED!")
    print()
    
    # Bounding box
    x, y, bw, bh = detection.bbox
    print(f"Bounding Box:")
    print(f"  Position: ({x}, {y})")
    print(f"  Size: {bw}x{bh} pixels")
    print(f"  Area: {detection.area} px² ({detection.normalized_area*100:.1f}% of frame)")
    
    # Center position
    print(f"\nCenter Position:")
    print(f"  Pixel: ({detection.center_x}, {detection.center_y})")
    print(f"  Frame center: ({w//2}, {h//2})")
    
    # Normalized X (for turning)
    norm_x = detection.normalized_x
    print(f"\nHorizontal Offset (normalized_x): {norm_x:.3f}")
    if norm_x < -0.1:
        print(f"  → Person is to the LEFT, robot should turn LEFT")
    elif norm_x > 0.1:
        print(f"  → Person is to the RIGHT, robot should turn RIGHT")
    else:
        print(f"  → Person is CENTERED ✓")
    
    # Area (for distance)
    area = detection.normalized_area
    print(f"\nRelative Size (normalized_area): {area:.4f}")
    if area > 0.25:
        print(f"  → Person is TOO CLOSE, robot should BACK UP")
    elif area > 0.08:
        print(f"  → Person is at good distance ✓")
    elif area > 0.02:
        print(f"  → Person is FAR, robot should MOVE FORWARD")
    else:
        print(f"  → Person is VERY FAR, robot should MOVE FORWARD faster")
    
    print(f"\nConfidence: {detection.confidence:.2f}")
    print("=" * 60)


def draw_detailed_overlay(frame: np.ndarray, detection: PersonDetection | None) -> np.ndarray:
    """Draw detailed debug overlay on frame."""
    output = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw frame center crosshair
    cx, cy = w // 2, h // 2
    cv2.line(output, (cx, 0), (cx, h), (100, 100, 100), 1)
    cv2.line(output, (0, cy), (w, cy), (100, 100, 100), 1)
    
    # Draw center zone (where we consider person "centered")
    deadzone = int(w * 0.1)  # 10% deadzone
    cv2.rectangle(output, (cx - deadzone, 0), (cx + deadzone, h), (50, 50, 50), 1)
    
    if detection is None:
        # No detection - show red border
        cv2.rectangle(output, (5, 5), (w-5, h-5), (0, 0, 255), 3)
        cv2.putText(output, "NO PERSON DETECTED", (w//2 - 150, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return output
    
    # Draw bounding box
    x, y, bw, bh = detection.bbox
    cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    
    # Draw center point
    cv2.circle(output, (detection.center_x, detection.center_y), 8, (0, 0, 255), -1)
    cv2.circle(output, (detection.center_x, detection.center_y), 10, (255, 255, 255), 2)
    
    # Draw line from frame center to person center
    cv2.line(output, (cx, cy), (detection.center_x, detection.center_y), (255, 255, 0), 2)
    
    # Info panel background
    cv2.rectangle(output, (5, 5), (300, 120), (0, 0, 0), -1)
    cv2.rectangle(output, (5, 5), (300, 120), (0, 255, 0), 1)
    
    # Info text
    y_offset = 25
    texts = [
        f"norm_x: {detection.normalized_x:+.3f}",
        f"norm_area: {detection.normalized_area:.4f}",
        f"bbox: {bw}x{bh}",
        f"confidence: {detection.confidence:.2f}",
    ]
    for text in texts:
        cv2.putText(output, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y_offset += 22
    
    # Direction indicator
    norm_x = detection.normalized_x
    if norm_x < -0.1:
        direction = "<< LEFT"
        color = (255, 165, 0)
    elif norm_x > 0.1:
        direction = "RIGHT >>"
        color = (255, 165, 0)
    else:
        direction = "CENTERED"
        color = (0, 255, 0)
    
    cv2.putText(output, direction, (w - 150, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Distance indicator
    area = detection.normalized_area
    if area > 0.25:
        dist = "TOO CLOSE!"
        color = (0, 0, 255)
    elif area > 0.08:
        dist = "GOOD DIST"
        color = (0, 255, 0)
    else:
        dist = "TOO FAR"
        color = (255, 165, 0)
    
    cv2.putText(output, dist, (w - 150, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return output


def run_snapshot_test(camera_index: int, use_mediapipe: bool):
    """Capture one frame, analyze, save results."""
    log("=" * 60)
    log("SNAPSHOT TEST")
    log("=" * 60)
    
    # Capture
    frame = capture_frame(camera_index)
    raw_path = save_debug_image(frame, "_raw")
    
    # Initialize detector
    log(f"Initializing detector (MediaPipe: {use_mediapipe and MEDIAPIPE_AVAILABLE})...")
    detector = PersonDetector(use_mediapipe=use_mediapipe)
    
    # Detect
    log("Running person detection...")
    start = time.time()
    detection = detector.detect(frame)
    elapsed = time.time() - start
    log(f"Detection took {elapsed*1000:.1f} ms")
    
    # Analyze
    analyze_detection(detection, frame.shape)
    
    # Draw and save annotated image
    annotated = draw_detailed_overlay(frame, detection)
    annotated_path = save_debug_image(annotated, "_annotated")
    
    print(f"\n📷 Raw image saved: {raw_path}")
    print(f"📷 Annotated image saved: {annotated_path}")
    
    # Show images
    print("\nPress any key to close windows...")
    cv2.imshow("Raw Frame", frame)
    cv2.imshow("Detection Result", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    detector.close()


def run_live_test(camera_index: int, use_mediapipe: bool):
    """Run live detection with full debug output."""
    log("=" * 60)
    log("LIVE DETECTION TEST")
    log("=" * 60)
    log("Controls:")
    log("  'q' - Quit")
    log("  's' - Save current frame")
    log("  SPACE - Pause/unpause")
    log("=" * 60)
    
    # Initialize
    log(f"Initializing detector (MediaPipe: {use_mediapipe and MEDIAPIPE_AVAILABLE})...")
    detector = PersonDetector(use_mediapipe=use_mediapipe)
    
    log(f"Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        log("Failed to open camera!", "ERROR")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    paused = False
    frame_count = 0
    fps_start = time.time()
    fps = 0
    
    log("Starting live detection... Press 'q' to quit")
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    log("Failed to read frame", "WARN")
                    continue
                
                # Detect
                detection = detector.detect(frame)
                
                # Draw overlay
                display = draw_detailed_overlay(frame, detection)
                
                # FPS counter
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start)
                    fps_start = time.time()
                
                cv2.putText(display, f"FPS: {fps:.1f}", (10, display.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Log periodically
                if frame_count % 60 == 0:
                    if detection:
                        log(f"Detection: x={detection.normalized_x:+.2f}, area={detection.normalized_area:.4f}")
                    else:
                        log("No detection")
            
            cv2.imshow("Live Detection Test", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_debug_image(frame, "_raw")
                save_debug_image(display, "_annotated")
                log("Saved current frame!")
            elif key == ord(' '):
                paused = not paused
                log(f"{'PAUSED' if paused else 'RESUMED'}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        log("Done!")


def run_image_test(image_path: str, use_mediapipe: bool):
    """Test detection on an existing image file."""
    log("=" * 60)
    log(f"IMAGE TEST: {image_path}")
    log("=" * 60)
    
    if not os.path.exists(image_path):
        log(f"Image not found: {image_path}", "ERROR")
        return
    
    frame = cv2.imread(image_path)
    if frame is None:
        log(f"Failed to read image: {image_path}", "ERROR")
        return
    
    h, w = frame.shape[:2]
    log(f"Image size: {w}x{h}")
    
    # Initialize detector
    log(f"Initializing detector (MediaPipe: {use_mediapipe and MEDIAPIPE_AVAILABLE})...")
    detector = PersonDetector(use_mediapipe=use_mediapipe)
    
    # Detect
    log("Running person detection...")
    start = time.time()
    detection = detector.detect(frame)
    elapsed = time.time() - start
    log(f"Detection took {elapsed*1000:.1f} ms")
    
    # Analyze
    analyze_detection(detection, frame.shape)
    
    # Draw and save annotated image
    annotated = draw_detailed_overlay(frame, detection)
    annotated_path = save_debug_image(annotated, "_annotated")
    
    print(f"\n📷 Annotated image saved: {annotated_path}")
    
    # Show
    print("\nPress any key to close...")
    cv2.imshow("Detection Result", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    detector.close()


def main():
    parser = argparse.ArgumentParser(description="Test person detection")
    parser.add_argument("--camera", "-c", type=int, default=0,
                       help="Camera index (default: 0)")
    parser.add_argument("--snapshot", action="store_true",
                       help="Take one snapshot and analyze")
    parser.add_argument("--image", "-i", type=str,
                       help="Test on existing image file")
    parser.add_argument("--no-mediapipe", action="store_true",
                       help="Use HOG instead of MediaPipe")
    parser.add_argument("--test-camera", action="store_true",
                       help="Just test if camera works")
    
    args = parser.parse_args()
    use_mediapipe = not args.no_mediapipe
    
    print()
    log("=" * 60)
    log("PERSON DETECTION TEST TOOL")
    log(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")
    log(f"Using MediaPipe: {use_mediapipe and MEDIAPIPE_AVAILABLE}")
    log("=" * 60)
    print()
    
    if args.test_camera:
        test_camera(args.camera)
    elif args.image:
        run_image_test(args.image, use_mediapipe)
    elif args.snapshot:
        run_snapshot_test(args.camera, use_mediapipe)
    else:
        run_live_test(args.camera, use_mediapipe)


if __name__ == "__main__":
    main()


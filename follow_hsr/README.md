# Human Following Robot for iRobot Create3

A ROS 2 package that enables the Create3 robot to detect and follow a human using computer vision with MediaPipe.

## Overview

The robot uses a USB webcam to detect humans via MediaPipe Pose detection, then drives toward the detected person. Simple and effective - when it sees you, it chases you!

## Features

- **Real-time Person Detection** - MediaPipe Pose (fast, works on Raspberry Pi)
- **Simple Follow Behavior** - Turns toward person + drives forward
- **Headless Operation** - Works over SSH with no display needed
- **Debug Image Saving** - Periodic snapshots to verify what the robot sees
- **Fallback Detection** - OpenCV HOG detector if MediaPipe unavailable

## Requirements

### Hardware
- iRobot Create3
- Raspberry Pi (or other Linux computer) connected to Create3
- USB Webcam

### Software
- ROS 2 Jazzy
- Python 3.10+
- OpenCV
- MediaPipe

## Installation

```bash
# 1. Clone/copy to your robot
cd ~/robot_ai

# 2. Create virtual environment (if not exists)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r follow_hsr/requirements.txt
```

## Quick Start

```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Activate venv
cd ~/robot_ai
source .venv/bin/activate

# Run the follower (headless mode with debug images)
python follow_hsr/follow_human.py --no-video --save-images
```

## Usage

### Basic Commands

```bash
# Headless mode (SSH) with debug images saved every 2 seconds
python follow_hsr/follow_human.py --no-video --save-images

# With display (if monitor connected)
python follow_hsr/follow_human.py

# Adjust speed
python follow_hsr/follow_human.py --no-video --max-linear-speed 0.15 --max-angular-speed 0.8

# Use different camera
python follow_hsr/follow_human.py --no-video --camera 1
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--camera`, `-c` | 0 | Camera device index |
| `--no-video` | False | Disable video window (for SSH/headless) |
| `--no-mediapipe` | False | Use OpenCV HOG instead of MediaPipe |
| `--save-images` | False | Save debug images periodically |
| `--save-interval` | 2.0 | Seconds between debug image saves |
| `--max-linear-speed` | 0.2 | Max forward speed (m/s) |
| `--max-angular-speed` | 1.0 | Max turning speed (rad/s) |

### Keyboard Controls (with display)

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `p` | Pause/Resume |
| `s` | Toggle search mode |
| `d` | Toggle video display |
| `SPACE` | Emergency stop |

## Testing Detection

Before running the full follower, test that detection works:

```bash
# Test camera
python follow_hsr/test_detection.py --test-camera

# Take one snapshot and analyze
python follow_hsr/test_detection.py --snapshot

# Test on existing image
python follow_hsr/test_detection.py --image path/to/photo.jpg
```

Debug images are saved to `follow_hsr/debug_output/`.

## How It Works

### Detection
1. Capture frame from USB webcam
2. Run MediaPipe Pose detection
3. Calculate bounding box around detected pose landmarks
4. Compute `normalized_x` (-1 to +1, where 0 = centered)
5. Compute `normalized_area` (larger = closer)

### Control
When a person is detected:
- **Turn**: Proportional control to center person in frame
- **Drive**: Always move forward (speed varies by distance)

```
Person to LEFT  → Turn LEFT  (positive angular velocity)
Person to RIGHT → Turn RIGHT (negative angular velocity)
Person detected → DRIVE FORWARD (always)
```

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ USB Camera  │ --> │ MediaPipe    │ --> │ Controller  │
│ (OpenCV)    │     │ Pose Detect  │     │ (Turn+Drive)│
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                    ┌─────────────┐     ┌────────▼──────┐
                    │ Create3     │ <-- │ /cmd_vel      │
                    │ Robot       │     │ (Twist msg)   │
                    └─────────────┘     └───────────────┘
```

## Files

| File | Description |
|------|-------------|
| `follow_human.py` | Main ROS 2 node - runs the follower |
| `detector.py` | Person detection (MediaPipe + HOG fallback) |
| `controller.py` | Motion control logic |
| `test_detection.py` | Standalone detection testing tool |
| `requirements.txt` | Python dependencies |

## Troubleshooting

### Robot doesn't move
1. **Check Create3 status** - Red blinking light = error/low battery
2. **Test cmd_vel directly**:
   ```bash
   ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1}}" --once
   ```
3. **Check for hazards**:
   ```bash
   ros2 topic echo /hazard_detection --once
   ```

### No person detected
1. Check camera: `python follow_hsr/test_detection.py --test-camera`
2. Check lighting - MediaPipe needs decent light
3. Stand 1-3 meters from camera, facing it
4. Check debug images in `follow_hsr/debug_output/`

### MediaPipe errors
- First run builds font cache (takes 30-60s on Pi)
- If crashes, try: `pip install --upgrade mediapipe`

### Display errors (Qt/xcb)
- Use `--no-video` flag for headless operation
- Debug images are saved to `debug_output/` folder

## Tuning

Edit `controller.py` `ControllerConfig` class:

```python
angular_kp = 0.8          # Turn aggressiveness (higher = faster turning)
max_angular_speed = 1.0   # Max turn speed (rad/s)
max_linear_speed = 0.2    # Max forward speed (m/s)
angular_deadzone = 0.1    # Don't turn if person nearly centered
```

## License

MIT

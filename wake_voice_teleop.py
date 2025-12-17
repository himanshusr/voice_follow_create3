#!/usr/bin/env python3
import os
import time
import threading
import random
import subprocess
from pathlib import Path
from typing import List, Dict

import pvporcupine
from pvrecorder import PvRecorder

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import whisper

from robot.intent import parse_intent
from robot.vision import describe_scene_with_gemini
from robot.tts import speak

# ================== CONFIG ==================

# Picovoice access key
ACCESS_KEY = (
    os.environ.get("PICOVOICE_ACCESS_KEY")
    or os.environ.get("PICO_ACCESS_KEY")
    or ""
).strip()

# USB mic ALSA device for arecord (we already tested this)
ARECORD_DEVICE = "plughw:0,0"

# PvRecorder device index for wake word (USB PnP Sound Device)
# From your logs, this is:
#   3: USB PnP Sound Device, USB Audio
WAKE_AUDIO_DEVICE_INDEX = 3

# Directory for temporary audio files
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
LAST_COMMAND_WAV = DATA_DIR / "last_command.wav"

# Gemini API key must be in env for describe_scene_with_gemini()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# Unknown-command responses
UNKNOWN_RESPONSES = [
    "Sorry, I didn't quite understand that.",
    "I didn't catch that. Could you say it again?",
    "I'm not sure what you meant. Please try again.",
]


# ================== WHISPER RECOGNIZER ==================

class WhisperRecognizer:
    """
    Simple Whisper-based recognizer that records with arecord, then transcribes.
    """

    def __init__(self, model_name: str = "tiny", sample_rate: int = 16000, arecord_device: str = "plughw:0,0"):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.arecord_device = arecord_device
        print(f"[ASR] Loading Whisper model '{model_name}'...")
        self.model = whisper.load_model(model_name)
        print("[ASR] Whisper model loaded.")

    def _record_once(self, duration_sec: float) -> bool:
        """
        Record a single utterance to LAST_COMMAND_WAV using arecord.
        Returns True on success, False on failure.
        """
        cmd = [
            "arecord",
            "-D", self.arecord_device,
            "-f", "S16_LE",
            "-r", str(self.sample_rate),
            "-c", "1",
            "-d", str(int(duration_sec)),  # must be int for arecord
            "-t", "wav",
            str(LAST_COMMAND_WAV),
        ]
        print(f"[ASR] Recording via arecord for {duration_sec} seconds...")
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ASR] arecord failed: {e}")
            return False
        except FileNotFoundError:
            print("[ASR] arecord not found. Is ALSA installed?")
            return False

    def listen_and_transcribe(self, max_seconds: float = 8.0) -> str:
        """
        Blocking: record once up to max_seconds, then transcribe with Whisper.
        Returns the recognized text (possibly empty string if nothing).
        """
        if not self._record_once(max_seconds):
            return ""

        if not LAST_COMMAND_WAV.exists():
            print("[ASR] No WAV file was recorded.")
            return ""

        print("[ASR] Running Whisper transcription...")
        try:
            result = self.model.transcribe(str(LAST_COMMAND_WAV), fp16=False)
            text = (result.get("text") or "").strip()
            print(f"[ASR] Transcript: '{text}'")
            return text
        except Exception as e:
            print(f"[ASR] Whisper transcription failed: {e}")
            return ""


# ================== CREATE3 TELEOP NODE ==================

class Create3VoiceTeleop(Node):
    """
    Simple ROS 2 node to send velocity commands to the iRobot Create3.
    """

    def __init__(self):
        super().__init__("create3_voice_teleop")
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Tunable speeds
        self.default_lin_speed = 0.20  # m/s
        self.default_ang_speed = 1.0   # rad/s

        print("[ROS] Create3VoiceTeleop ready, publishing to /cmd_vel.")

    # ---- Low-level movement helpers ----

    def stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def drive_forward(self, distance_m: float, speed: float | None = None):
        speed = speed or self.default_lin_speed
        speed = max(0.0, min(speed, 0.22))  # Create3 limit
        duration = abs(distance_m) / max(speed, 1e-3)

        print(f"[MOVE] Forward {distance_m:.2f} m at {speed:.2f} m/s (duration ~{duration:.2f}s).")
        twist = Twist()
        twist.linear.x = speed * (1 if distance_m >= 0 else -1)

        end_time = time.time() + duration
        while time.time() < end_time and rclpy.ok():
            self.cmd_pub.publish(twist)
            time.sleep(0.05)

        self.stop()

    def turn_in_place(self, degrees: float, ang_speed: float | None = None):
        ang_speed = ang_speed or self.default_ang_speed
        ang_speed = max(0.1, min(ang_speed, 2.0))
        radians = abs(degrees) * 3.14159265 / 180.0
        duration = radians / ang_speed

        direction = 1.0 if degrees >= 0 else -1.0
        print(f"[TURN] {degrees:.1f} deg at {ang_speed:.2f} rad/s (duration ~{duration:.2f}s).")

        twist = Twist()
        twist.angular.z = direction * ang_speed

        end_time = time.time() + duration
        while time.time() < end_time and rclpy.ok():
            self.cmd_pub.publish(twist)
            time.sleep(0.05)

        self.stop()

    # ---- High-level sequence execution ----

    def execute_movement_sequence(self, actions: List[Dict]):
        """
        Execute a parsed action list from parse_intent().
        Each action is a dict like:
          { "type": "move", "direction": "forward", "distance_m": 1.0, "speed": 0.2 }
          { "type": "turn", "direction": "left", "degrees": 90 }
          { "type": "stop" }
        """
        print("[MOVE SEQUENCE]")
        for i, action in enumerate(actions, start=1):
            print(f"  Step {i}: {action}")

            atype = action.get("type")
            if atype == "move":
                direction = action.get("direction", "forward")
                dist = float(action.get("distance_m") or 0.0)
                if direction == "backward":
                    dist = -abs(dist)
                else:
                    dist = abs(dist)
                speed = action.get("speed") or self.default_lin_speed
                self.drive_forward(dist, speed=float(speed))

            elif atype == "turn":
                direction = action.get("direction", "right")
                deg = float(action.get("degrees") or 0.0)
                if direction == "left":
                    deg = abs(deg)
                else:
                    deg = -abs(deg)
                self.turn_in_place(deg)

            elif atype == "stop":
                self.stop()

    # ---- Scene description ----

    def describe_surroundings(self) -> str:
        """
        Capture an image from /dev/video0 and send to Gemini for description.
        This delegates to robot.vision.describe_scene_with_gemini().
        """
        if not GEMINI_API_KEY:
            print("[VISION] GEMINI_API_KEY not set; cannot describe surroundings.")
            return "I'm having trouble seeing right now."

        try:
            desc = describe_scene_with_gemini(
                api_key=GEMINI_API_KEY,
                camera_index=0,
            )
            return desc or "I'm not sure what I see."
        except Exception as e:
            print(f"[VISION ERROR] {e}")
            return "I'm having trouble seeing right now."


# ================== INTENT HANDLING ==================

def format_actions_for_speech(actions: List[Dict]) -> str:
    """
    Turn a parsed movement sequence into a natural language summary.
    """
    parts: List[str] = []

    for action in actions:
        atype = action.get("type")

        if atype == "move":
            direction = action.get("direction", "forward")
            dist = action.get("distance_m")
            if dist is not None:
                try:
                    dist = float(dist)
                    if float(dist).is_integer():
                        dist_str = str(int(dist))
                    else:
                        dist_str = f"{dist:g}"
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
                    deg = float(deg)
                    if float(deg).is_integer():
                        deg_str = str(int(deg))
                    else:
                        deg_str = f"{deg:g}"
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


def handle_command(node: Create3VoiceTeleop, text: str) -> bool:
    """
    Parse and execute a command string.
    Returns True if recognized, False if unknown.
    """
    intent = parse_intent(text)
    kind = intent.get("intent", "unknown")
    print(f"[INTENT] {intent}")

    if kind == "movement_sequence":
        actions = intent.get("actions", [])
        if not actions:
            return False
        summary = format_actions_for_speech(actions)
        speak(f"Okay, {summary}.")
        node.execute_movement_sequence(actions)
        return True

    if kind == "describe_surroundings":
        result = {}
        done = threading.Event()

        def worker():
            desc = node.describe_surroundings()
            result["desc"] = desc
            done.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        speak("One second, I'm analyzing what I see.")
        done.wait()
        desc = result.get("desc") or "I'm not sure what I see."
        speak(desc)
        return True

    if kind == "stop":
        node.stop()
        speak("Stopping all movement.")
        return True

    if kind == "shutdown":
        speak("Shutting down the voice interface.")
        return True

    return False


# ================== WAKE WORD + COMMAND LOOP (PORCUPINE) ==================

def main():
    if not ACCESS_KEY or ACCESS_KEY == "REPLACE_ME_WITH_YOUR_ACCESS_KEY":
        print("[ERROR] Picovoice ACCESS_KEY not set. Set PICOVOICE_ACCESS_KEY or PICO_ACCESS_KEY in your env.")
        return

    rclpy.init()
    node = Create3VoiceTeleop()

    # Whisper recognizer for commands
    recognizer = WhisperRecognizer(
        model_name="tiny",
        sample_rate=16000,
        arecord_device=ARECORD_DEVICE,
    )

    # Initialize Porcupine once
    print("[WAKE] Initializing Porcupine...")
    porcupine = pvporcupine.create(
        access_key=ACCESS_KEY,
        keywords=["computer"],
    )

    # Show available devices for debug
    devices = PvRecorder.get_available_devices()
    print("Available audio devices for wake word:")
    for i, name in enumerate(devices):
        print(f"  {i}: {name}")

    if WAKE_AUDIO_DEVICE_INDEX < 0 or WAKE_AUDIO_DEVICE_INDEX >= len(devices):
        print(f"[WAKE] Invalid WAKE_AUDIO_DEVICE_INDEX={WAKE_AUDIO_DEVICE_INDEX}.")
        porcupine.delete()
        rclpy.shutdown()
        return

    print("\nRobot is idle and listening for 'computer'.")
    print("Say 'computer' to wake the robot. Say 'shut down' to exit.\n")

    running = True

    try:
        while rclpy.ok() and running:
            # --- WAKE PHASE: Porcupine + PvRecorder ---
            print("[IDLE] Setting up mic for wake-word listening...")
            recorder = PvRecorder(
                device_index=WAKE_AUDIO_DEVICE_INDEX,
                frame_length=porcupine.frame_length,
            )

            try:
                recorder.start()
                print(f"[IDLE] Listening for 'computer' on device index {WAKE_AUDIO_DEVICE_INDEX} ({devices[WAKE_AUDIO_DEVICE_INDEX]})...")

                wake_detected = False
                while rclpy.ok() and running:
                    rclpy.spin_once(node, timeout_sec=0.01)
                    pcm = recorder.read()
                    result = porcupine.process(pcm)
                    if result >= 0:
                        print("[WAKE] Wake word detected!")
                        wake_detected = True
                        break

            except KeyboardInterrupt:
                print("\n[EXIT] KeyboardInterrupt during wake listening.")
                running = False
            finally:
                try:
                    recorder.stop()
                except Exception:
                    pass
                recorder.delete()

            if not running or not rclpy.ok():
                break

            if not wake_detected:
                # E.g. broke out because of shutdown/interrupt
                continue

            # --- COMMAND PHASE: arecord + Whisper ---
            # Small pause to ensure ALSA releases the device fully
            time.sleep(0.2)

            speak("Yes?")

            text = recognizer.listen_and_transcribe(max_seconds=8.0)

            if not text:
                print("[ASR] No command heard after wake.")
                speak("I did not hear a command.")
            else:
                recognized = handle_command(node, text)
                if not recognized:
                    reply = random.choice(UNKNOWN_RESPONSES)
                    speak(reply)

                # If the user's intent was shutdown, stop the outer loop
                intent = parse_intent(text)
                if intent.get("intent") == "shutdown":
                    running = False
                    break

            print("\n[IDLE] Returning to wake-word listening...\n")

    finally:
        porcupine.delete()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

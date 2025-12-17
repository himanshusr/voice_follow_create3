import os
import shlex
import subprocess


def speak(text: str, rate: int = 170):
    """
    On the Pi: if no speaker is available, just print the text.
    Later we can reroute this to the Create3's speaker via ROS.
    """
    if not text:
        return

    # TEMP: comment out actual audio to avoid ALSA errors on headless Pi
    print(f"[TTS] {text}")

    # If you later plug in a speaker and fix ALSA, you can uncomment this:
    # cmd = ["espeak-ng", "-s", str(rate), text]
    # try:
    #     subprocess.run(cmd, check=True)
    # except FileNotFoundError:
    #     print("[TTS ERROR] espeak-ng not found.")
    # except subprocess.CalledProcessError as e:
    #     print(f"[TTS ERROR] espeak-ng failed: {e}")

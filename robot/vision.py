#!/usr/bin/env python3
import os
from pathlib import Path

import cv2
import google.generativeai as genai


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
IMAGE_PATH = DATA_DIR / "scene.jpg"

# Choose a good multimodal model (from your ListModels output)
GEMINI_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")


def _configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
    genai.configure(api_key=api_key)


def capture_frame(device_index: int = 0, warmup_frames: int = 5) -> Path:
    """
    Capture a single frame from the given camera index and save to IMAGE_PATH.
    Returns the Path to the image file.
    """
    # Try with V4L2 backend first (better on Linux), then fallback
    cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {device_index}")

    # Warm-up frames so exposure/auto-settings settle
    for _ in range(warmup_frames):
        ret, _ = cap.read()
        if not ret:
            break

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame from camera.")

    # Save as JPEG
    ok = cv2.imwrite(str(IMAGE_PATH), frame)
    if not ok:
        raise RuntimeError(f"Failed to write image to {IMAGE_PATH}")

    return IMAGE_PATH


def describe_scene_with_gemini(
    device_index: int = 0,
    extra_instruction: str | None = None,
) -> str:
    """
    Capture an image from the camera and ask Gemini to describe it.
    Returns the description text, or raises an exception on failure.
    """
    _configure_gemini()

    base_prompt = (
        "You are a mobile robot with a forward-facing camera. "
        "Describe what you see in front of you as if you are talking to a human user. "
        "Mention important objects, distances in rough terms (like 'near', 'far', "
        "'to your left/right'), and anything that might affect navigation."
    )

    if extra_instruction:
        prompt = base_prompt + " " + extra_instruction
    else:
        prompt = base_prompt

    # Capture the image
    img_path = capture_frame(device_index=device_index)

    # Read image bytes
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    model = genai.GenerativeModel(GEMINI_MODEL)

    try:
        response = model.generate_content(
            [
                prompt,
                {"mime_type": "image/jpeg", "data": img_bytes},
            ],
            # keep it safe-ish on a small Pi
            request_options={"timeout": 60},
        )
    except Exception as e:
        raise RuntimeError(f"Gemini generate_content failed: {e}")

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty description.")

    return text

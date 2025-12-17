#!/usr/bin/env python3
"""
Person detection module for human-following robot.

Supports two backends:
  1. MediaPipe (recommended) - faster and more accurate
  2. OpenCV HOG - fallback if MediaPipe unavailable
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[DETECTOR] MediaPipe not available, will use OpenCV HOG fallback")


@dataclass
class PersonDetection:
    """Represents a detected person in the frame."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center_x: int  # center x coordinate
    center_y: int  # center y coordinate
    area: int  # bbox area (proxy for distance)
    confidence: float  # detection confidence (0-1)

    @property
    def normalized_x(self) -> float:
        """X position normalized to [-1, 1] where 0 is center of frame."""
        return self._norm_x

    @normalized_x.setter
    def normalized_x(self, value: float):
        self._norm_x = value

    @property
    def normalized_area(self) -> float:
        """Area normalized relative to frame size."""
        return self._norm_area

    @normalized_area.setter
    def normalized_area(self, value: float):
        self._norm_area = value


class PersonDetector:
    """
    Detects people in camera frames.
    Uses MediaPipe if available, falls back to OpenCV HOG.
    """

    def __init__(
        self,
        use_mediapipe: bool = True,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0,  # 0 = short-range (2m), 1 = full-range (5m)
    ):
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        self.min_confidence = min_detection_confidence

        if self.use_mediapipe:
            self._init_mediapipe(model_selection)
        else:
            self._init_hog()

        self.frame_width = 640
        self.frame_height = 480

    def _init_mediapipe(self, model_selection: int):
        """Initialize MediaPipe pose/detection."""
        print("[DETECTOR] Initializing MediaPipe person detector...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 0=lite, 1=full, 2=heavy
            min_detection_confidence=self.min_confidence,
            min_tracking_confidence=0.5,
        )
        print("[DETECTOR] MediaPipe ready.")

    def _init_hog(self):
        """Initialize OpenCV HOG pedestrian detector."""
        print("[DETECTOR] Initializing OpenCV HOG detector...")
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print("[DETECTOR] HOG detector ready.")

    def detect(self, frame: np.ndarray) -> Optional[PersonDetection]:
        """
        Detect a person in the frame.
        Returns the most prominent (largest) person detection, or None.
        """
        self.frame_height, self.frame_width = frame.shape[:2]

        if self.use_mediapipe:
            return self._detect_mediapipe(frame)
        else:
            return self._detect_hog(frame)

    def _detect_mediapipe(self, frame: np.ndarray) -> Optional[PersonDetection]:
        """Detect person using MediaPipe Pose."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        # Get bounding box from pose landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Get all visible landmark coordinates
        x_coords = []
        y_coords = []
        for lm in landmarks:
            if lm.visibility > 0.5:
                x_coords.append(lm.x * self.frame_width)
                y_coords.append(lm.y * self.frame_height)

        if len(x_coords) < 5:  # Need enough landmarks
            return None

        # Calculate bounding box
        x_min = int(max(0, min(x_coords) - 20))
        x_max = int(min(self.frame_width, max(x_coords) + 20))
        y_min = int(max(0, min(y_coords) - 20))
        y_max = int(min(self.frame_height, max(y_coords) + 20))

        width = x_max - x_min
        height = y_max - y_min
        center_x = x_min + width // 2
        center_y = y_min + height // 2
        area = width * height

        detection = PersonDetection(
            bbox=(x_min, y_min, width, height),
            center_x=center_x,
            center_y=center_y,
            area=area,
            confidence=0.9,  # MediaPipe doesn't give overall confidence
        )

        # Normalize values
        detection.normalized_x = (center_x - self.frame_width / 2) / (self.frame_width / 2)
        detection.normalized_area = area / (self.frame_width * self.frame_height)

        return detection

    def _detect_hog(self, frame: np.ndarray) -> Optional[PersonDetection]:
        """Detect person using OpenCV HOG."""
        # Resize for faster detection
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # Detect people
        boxes, weights = self.hog.detectMultiScale(
            small_frame,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05,
        )

        if len(boxes) == 0:
            return None

        # Scale boxes back up and find largest
        best_idx = 0
        best_area = 0
        for i, (x, y, w, h) in enumerate(boxes):
            area = w * h
            if area > best_area:
                best_area = area
                best_idx = i

        x, y, w, h = boxes[best_idx]
        # Scale back to original size
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

        center_x = x + w // 2
        center_y = y + h // 2
        area = w * h

        detection = PersonDetection(
            bbox=(x, y, w, h),
            center_x=center_x,
            center_y=center_y,
            area=area,
            confidence=float(weights[best_idx]),
        )

        # Normalize values
        detection.normalized_x = (center_x - self.frame_width / 2) / (self.frame_width / 2)
        detection.normalized_area = area / (self.frame_width * self.frame_height)

        return detection

    def draw_detection(self, frame: np.ndarray, detection: PersonDetection) -> np.ndarray:
        """Draw bounding box and info on frame for debugging."""
        x, y, w, h = detection.bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw center point
        cv2.circle(frame, (detection.center_x, detection.center_y), 5, (0, 0, 255), -1)
        
        # Draw frame center line
        cx = self.frame_width // 2
        cv2.line(frame, (cx, 0), (cx, self.frame_height), (255, 0, 0), 1)
        
        # Draw info text
        info = f"x: {detection.normalized_x:.2f}, area: {detection.normalized_area:.3f}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def close(self):
        """Release resources."""
        if self.use_mediapipe and hasattr(self, 'pose'):
            self.pose.close()


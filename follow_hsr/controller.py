#!/usr/bin/env python3
"""
Motion controller for human-following robot.

Uses proportional control to:
  - Turn toward the detected person (angular velocity)
  - Move forward/backward to maintain target distance (linear velocity)
"""

from dataclasses import dataclass
from typing import Tuple
from detector import PersonDetection


@dataclass
class ControllerConfig:
    """Configuration for the follow controller."""
    
    # Angular control (turning)
    angular_kp: float = 0.8          # Proportional gain for turning
    max_angular_speed: float = 1.0   # rad/s (Create3 max ~1.9)
    angular_deadzone: float = 0.1    # Don't turn if person nearly centered
    
    # Linear control (forward/backward)
    linear_kp: float = 0.5           # Proportional gain for linear motion
    max_linear_speed: float = 0.2    # m/s (Create3 max ~0.306)
    min_linear_speed: float = 0.05   # Minimum speed when moving
    
    # Target distance (based on normalized bbox area)
    target_area: float = 0.08        # Target bbox area (bigger = closer)
    area_tolerance: float = 0.02     # Don't move if within tolerance
    
    # Safety
    too_close_area: float = 0.25     # Stop/backup if person too close
    lost_timeout: float = 2.0        # Seconds before stopping when person lost


class FollowController:
    """
    Simple controller for following a person.
    
    When person detected: turn toward them + drive forward.
    """
    
    def __init__(self, config: ControllerConfig = None):
        self.config = config or ControllerConfig()
        self._lost_time = 0.0
    
    def compute(
        self,
        detection: PersonDetection | None,
        dt: float = 0.1,
    ) -> Tuple[float, float]:
        """
        Compute velocity commands based on person detection.
        
        Simple logic:
        - Person detected? Turn toward them and DRIVE FORWARD.
        - No person? Stop.
        """
        if detection is None:
            self._lost_time += dt
            return (0.0, 0.0)
        
        # Person found - reset lost timer
        self._lost_time = 0.0
        
        # --- Angular velocity (turning toward person) ---
        # Positive normalized_x = person is to the right
        # Negative angular_vel = turn right
        x_error = detection.normalized_x
        
        if abs(x_error) < self.config.angular_deadzone:
            angular_vel = 0.0
        else:
            angular_vel = -self.config.angular_kp * x_error
            # Clamp to max speed
            angular_vel = max(-self.config.max_angular_speed,
                            min(self.config.max_angular_speed, angular_vel))
        
        # --- Linear velocity: ALWAYS FORWARD when person detected ---
        # Go faster when person is far, slower when close
        area = detection.normalized_area
        
        if area > 0.4:
            # Very close - slow down a bit but still move forward
            linear_vel = self.config.max_linear_speed * 0.3
        elif area > 0.2:
            # Close - medium speed
            linear_vel = self.config.max_linear_speed * 0.5
        else:
            # Far - full speed ahead!
            linear_vel = self.config.max_linear_speed
        
        return (linear_vel, angular_vel)
    
    def reset(self):
        """Reset controller state."""
        self._lost_time = 0.0


class SearchController:
    """
    Simple controller for searching when person is lost.
    Rotates slowly to scan the environment.
    """
    
    def __init__(self, search_speed: float = 0.3):
        self.search_speed = search_speed
        self._direction = 1  # 1 = left, -1 = right
        self._search_time = 0.0
        self._switch_interval = 5.0  # Switch direction every N seconds
    
    def compute(self, dt: float) -> Tuple[float, float]:
        """Return velocity commands for searching."""
        self._search_time += dt
        
        if self._search_time > self._switch_interval:
            self._direction *= -1
            self._search_time = 0.0
        
        return (0.0, self._direction * self.search_speed)
    
    def reset(self):
        """Reset search state."""
        self._search_time = 0.0
        self._direction = 1


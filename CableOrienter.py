# MIT License

# Copyright (c) 2025 Abdelrahman Mahmoud and Rodolfo Verde, Technology Transfer Center Kitzingen 

# See LICENSE file in the root of this repository.




import math
from enum import Enum
from typing import Tuple, List

import cv2
import numpy as np
from ultralytics import YOLO


# Enum class to define the different cable types
class CableType(Enum):
    YELLOW = 0


def calculate_angle_correction(corners: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
                               use_min=True) -> float:
    """Calculate the minimum angle correction based on the four corners of the cable shape."""
    top_left, top_right, bottom_left, bottom_right = np.array(corners)

    angle_left = math.degrees(math.atan2(bottom_left[1] - top_left[1], bottom_left[0] - top_left[0]))
    angle_right = math.degrees(math.atan2(bottom_right[1] - top_right[1], bottom_right[0] - top_right[0]))

    correction = 90 - (min(angle_left, angle_right) if use_min else max(angle_left, angle_right))
    return np.deg2rad(correction)


def get_approx_polly(mask: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Approximate the convex hull of a mask and return the four corner points of the cable shape."""
    mask = np.array(mask, dtype=np.int32)
    hull = cv2.convexHull(mask)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    iterations = 0
    max_iterations = 10
    
    while len(approx) != 4 and iterations < max_iterations:
        if len(approx) > 4:
            epsilon *= 1.2  # Increase epsilon to get fewer points
        else:
            epsilon *= 0.8  # Decrease epsilon to get more points
        
        approx = cv2.approxPolyDP(hull, epsilon, True)
        iterations += 1
    
    if len(approx) != 4:
        if len(approx) > 4:
            rect = cv2.minAreaRect(hull)
            approx = np.array(cv2.boxPoints(rect), dtype=np.int32)
        else:
            raise ValueError(f"Could not approximate to 4 points (got {len(approx)} points)")

    approx = approx.reshape(4, 2)
    sorted_pts = sorted(approx, key=lambda p: (p[1], p[0]))
    top_left, top_right = sorted(sorted_pts[:2], key=lambda p: p[0])
    bottom_left, bottom_right = sorted(sorted_pts[2:], key=lambda p: p[0])

    # Convert numpy arrays to explicit tuples with exactly two integers
    top_left = (int(top_left[0]), int(top_left[1]))
    top_right = (int(top_right[0]), int(top_right[1]))
    bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

    return top_left, top_right, bottom_left, bottom_right


class CableOrienter:
    """YOOO (You Only Orient Once) implementation for vision-driven cable orientation."""
    
    def __init__(self, keypoint_model_path='./models/yellow_kp.pt', 
                 segmentation_model_path='./models/yellow_seg.pt'):
        """Initializes the YOOO system with models for keypoint detection and segmentation."""
        self.models = {}
        
        try:
            self.models['kp'] = YOLO(keypoint_model_path)
            self.models['seg'] = YOLO(segmentation_model_path)
        
        except Exception as e:
            print(f"\nError loading models: {str(e)}")
            print("Make sure the model files exist in the specified paths:")
            print(f"  - {keypoint_model_path}")
            print(f"  - {segmentation_model_path}")
            raise
    
    def determine_roll_angle(self, image: np.ndarray) -> float:
        """Determines the roll correction angle using segmentation-based detection."""
        result = self.models['seg'](image, verbose=False)[0].cpu()
        
        if result.masks is not None and len(result.masks.xy) > 0:
            cable = get_approx_polly(result.masks.xy[0])
            return calculate_angle_correction(cable)
        else:
            raise ValueError("No cable detected in the image")
    
    def determine_pitch_angle(self, image: np.ndarray) -> float:
        """Determines the pitch correction angle based on keypoint detection."""
        result = self.models['kp'](image, verbose=False)[0].cpu()
        
        kps = result.keypoints
        if len(kps) > 0:
            keypoints = kps.xy.numpy()[0]
            
            x_left, y_left = keypoints[3]
            x_right, y_right = keypoints[2]
            
            delta_y = y_right - y_left
            delta_x = x_right - x_left
            
            detected_angle = math.degrees(math.atan2(delta_y, delta_x))
            angle_correction = 90 - np.abs(detected_angle)
            
            return np.deg2rad(angle_correction)
        else:
            raise ValueError("No keypoints detected in the image")
    
    def calculate_weighted_roll_angles(self, angles: List[float]) -> float:
        """Calculate weighted average of roll angles. Smaller angles get higher weights."""
        if not angles:
            return 0.0
        
        abs_angles = np.abs(np.array(angles))
        weights = 1.0 / (abs_angles + 1.0)
        weights = weights / np.sum(weights)
        weighted_angle = np.sum(np.array(angles) * weights)
        
        return np.deg2rad(weighted_angle)

    def get_detection_data(self, image: np.ndarray, detection_type: str):
        """Returns the raw detection data without visualization."""
        if detection_type == 'kp':
            # Run keypoint detection
            result = self.models['kp'](image, verbose=False)[0].cpu()
            
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.numpy()[0]
                return {'keypoints': keypoints}
            
        elif detection_type == 'seg':
            # Run segmentation
            result = self.models['seg'](image, verbose=False)[0].cpu()
            
            if result.masks is not None and len(result.masks.xy) > 0:
                mask = result.masks.xy[0]
                try:
                    corners = get_approx_polly(mask)
                    return {'mask': mask, 'corners': corners}
                except Exception:
                    return {'mask': mask}
                    
        return {}

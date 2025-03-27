import math
import time
from enum import Enum
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# Enum class to define the different cable types
class CableType(Enum):
    YELLOW = 0


def calculate_angle_correction(corners: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
                               use_min=True) -> float:
    """
    Calculate the minimum angle correction based on the four corners of the cable shape.

    Parameters:
    - corners: List of 4 tuples representing the (x, y) coordinates of the corners.
    - use_min: If True, returns the minimum angle; if False, returns the maximum angle.

    Returns:
    - The angle correction in radians.
    """
    # Extracting the four corners
    top_left, top_right, bottom_left, bottom_right = np.array(corners)

    # Compute the angles using arctan2 function (degrees)
    angle_left = math.degrees(math.atan2(bottom_left[1] - top_left[1], bottom_left[0] - top_left[0]))
    angle_right = math.degrees(math.atan2(bottom_right[1] - top_right[1], bottom_right[0] - top_right[0]))

    # Calculate the angle difference and apply a correction factor
    correction = 90 - (min(angle_left, angle_right) if use_min else max(angle_left, angle_right))
    return np.deg2rad(correction)


def get_approx_polly(mask: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Approximate the convex hull of a mask and return the four corner points of the cable shape.

    Parameters:
    - mask: The binary mask representing the cable region.

    Returns:
    - A tuple containing four points (top-left, top-right, bottom-left, bottom-right).
    """
    # Convert the mask to a numpy array of integer type
    mask = np.array(mask, dtype=np.int32)

    # Get the convex hull of the mask
    hull = cv2.convexHull(mask)

    # Start with an initial epsilon value for approximation
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # Try to get exactly 4 points using adaptive epsilon
    iterations = 0
    max_iterations = 10
    
    while len(approx) != 4 and iterations < max_iterations:
        if len(approx) > 4:
            epsilon *= 1.2  # Increase epsilon to get fewer points
        else:
            epsilon *= 0.8  # Decrease epsilon to get more points
        
        approx = cv2.approxPolyDP(hull, epsilon, True)
        iterations += 1
    
    # Handle cases where we couldn't get exactly 4 points
    if len(approx) != 4:
        if len(approx) > 4:
            # Use minimum area rectangle as fallback
            rect = cv2.minAreaRect(hull)
            approx = np.array(cv2.boxPoints(rect), dtype=np.int32)
        else:
            raise ValueError(f"Could not approximate to 4 points (got {len(approx)} points)")

    # Reshape and sort the points
    approx = approx.reshape(4, 2)
    
    # Sort points by y-coordinate (top-to-bottom)
    sorted_pts = sorted(approx, key=lambda p: (p[1], p[0]))

    # Identify corners (top-left, top-right, bottom-left, bottom-right)
    top_left, top_right = sorted(sorted_pts[:2], key=lambda p: p[0])
    bottom_left, bottom_right = sorted(sorted_pts[2:], key=lambda p: p[0])

    return top_left, top_right, bottom_left, bottom_right


class CableOrienter:
    """
    Utility class to handle cable orientation tasks, such as bending and pitch detection.
    Uses YOLO models to perform keypoint and segmentation detection on cable images.
    """
    # Paths to pre-trained YOLO models for keypoint and segmentation detection
    YELLOW_MODELS_PATH = {
        'kp': './models/yellow_kp.pt',  # Keypoint model path
        'seg': './models/yellow_seg.pt',  # Segmentation model path
    }

    def __init__(self, show_progress=False):
        """
        Initializes the utility by loading the YOLO models for keypoints and segmentation.
        
        Parameters:
        - show_progress: If True, displays a loading progress indicator
        """
        self.yellow_models = {}
        
        try:
            # Load keypoint model
            if show_progress:
                print("Loading keypoint model... ", end='', flush=True)
                start_time = time.time()
                
            self.yellow_models['kp'] = YOLO(self.YELLOW_MODELS_PATH['kp'])
            
            if show_progress:
                print(f"Done! ({time.time() - start_time:.2f}s)")
                print("Loading segmentation model... ", end='', flush=True)
                start_time = time.time()
            
            # Load segmentation model
            self.yellow_models['seg'] = YOLO(self.YELLOW_MODELS_PATH['seg'])
            
            if show_progress:
                print(f"Done! ({time.time() - start_time:.2f}s)")
        
        except Exception as e:
            print(f"\nError loading models: {str(e)}")
            print("Make sure the model files exist in the ./models/ directory:")
            print(f"  - {self.YELLOW_MODELS_PATH['kp']}")
            print(f"  - {self.YELLOW_MODELS_PATH['seg']}")
            raise

    def determine_bending(self, image: np.ndarray, cable_type: CableType, use_min=True) -> float:
        """
        Determines the bending correction for the cable based on segmentation results.

        Parameters:
        - image: The input image containing the cable.
        - cable_type: The type of cable (affects the model used).
        - use_min: If True, uses the minimum angle correction.

        Returns:
        - The angle correction for the bending in radians.
        """
        # Perform the detection using the segmentation model for yellow cables
        if cable_type == CableType.YELLOW:
            result = self.yellow_models['seg'](image, verbose=False)[0].cpu()
        else:
            raise ValueError(f"Cable type {cable_type.name} is not currently supported")

        # Extract the cable shape from the result's mask
        if result.masks is not None and len(result.masks.xy) > 0:
            cable = get_approx_polly(result.masks.xy[0])
            # Calculate the angle correction for bending
            return calculate_angle_correction(cable, use_min=use_min)
        else:
            raise ValueError("No cable detected in the image")

    def determine_pitch(self, image: np.ndarray, cable_type: CableType) -> float:
        """
        Determines the pitch correction for the cable based on keypoint detection.

        Parameters:
        - image: The input image containing the cable.
        - cable_type: The type of cable (affects the model used).

        Returns:
        - The angle correction for the pitch in radians.
        """
        # Perform the detection using the keypoint model for yellow cables
        if cable_type == CableType.YELLOW:
            result = self.yellow_models['kp'](image, verbose=False)[0].cpu()
        else:
            raise ValueError(f"Cable type {cable_type.name} is not currently supported")

        # Extract key points from the result
        kps = result.keypoints
        if len(kps) > 0:
            keypoints = kps.xy.numpy()[0]
                
            x_left, y_left = keypoints[3]
            x_right, y_right = keypoints[2]

            # Calculate the angle between the gradient and the X-axis
            delta_y = y_right - y_left
            delta_x = x_right - x_left

            detected_angle = math.degrees(math.atan2(delta_y, delta_x))

            # Calculate the angle correction needed to make the cable straight
            angle_correction = 90 - np.abs(detected_angle)

            return np.deg2rad(angle_correction)
        else:
            raise ValueError("No keypoints detected in the image")

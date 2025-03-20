from enum import Enum
from ultralytics import YOLO
import numpy as np
import cv2
import math
from typing import Tuple, List, Optional


# Enum class to define the different cable types
class CableType(Enum):
    CALIBRATION = 0  # Calibration with blue cable
    BLACKT1 = 1
    BLACKT2 = 2
    YELLOW_LOWER = 3
    YELLOW_UPPER = 4
    YELLOW2_LOWER = 5
    YELLOW2_UPPER = 6
    YELLOWT1 = 7
    YELLOWT2 = 8
    YELLOW2T1 = 9
    YELLOW2T2 = 10


def calculate_angle_correction(corners: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
                               use_min=True) -> float:
    """
    Calculate the minimum angle correction based on the four corners of the cable shape.

    Parameters:
    - corners: List of 4 tuples representing the (x, y) coordinates of the corners.

    Returns:
    - The angle correction in radians.
    """
    # Extracting the four corners
    top_left, top_right, bottom_left, bottom_right = np.array(corners)

    # Compute the angles using arctan2 function (degrees)
    angle_left = math.degrees(math.atan2(bottom_left[1] - top_left[1], bottom_left[0] - top_left[0]))
    angle_right = math.degrees(math.atan2(bottom_right[1] - top_right[1], bottom_right[0] - top_right[0]))

    # Calculate the smaller angle difference and apply a correction factor
    smaller = 90 - (min(angle_left, angle_right) if use_min else max(angle_left, angle_right))
    return np.deg2rad(smaller)


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

    # Approximate the contour with 4 points using a precision epsilon value
    epsilon = 0.02 * cv2.arcLength(hull, True)  # Adjust precision as needed
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # Force the approximation to have exactly 4 points
    if len(approx) > 4:
        approx = cv2.approxPolyDP(hull, 0.05 * cv2.arcLength(hull, True), True)

    # If the approximation has 4 points, sort them and return the corners
    if len(approx) == 4:
        approx = approx.reshape(4, 2)
        sorted_pts = sorted(approx, key=lambda p: (p[1], p[0]))  # Sort by y first, then x

        # Identify and sort the corners (top-left, top-right, bottom-left, bottom-right)
        top_left, top_right = sorted(sorted_pts[:2], key=lambda p: p[0])  # Sort left-to-right
        bottom_left, bottom_right = sorted(sorted_pts[2:], key=lambda p: p[0])

        return top_left, top_right, bottom_left, bottom_right
    else:
        raise ValueError("No valid cable shape detected in the image")


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

    def __init__(self):
        """
        Initializes the utility by loading the YOLO models for keypoints and segmentation.
        """
        self.yellow_models = {
            'kp': YOLO(self.YELLOW_MODELS_PATH['kp']),
            'seg': YOLO(self.YELLOW_MODELS_PATH['seg'])
        }

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
        if cable_type in [CableType.YELLOWT1, CableType.YELLOWT2, CableType.YELLOW2T1, CableType.YELLOW2T2]:
            result = self.yellow_models['seg'](image, verbose=False)[0].cpu()
        else:
            raise ValueError(f"Invalid cable type: {cable_type}")

        # Extract the cable shape from the result's mask
        if result.masks is not None:
            cable = get_approx_polly(result.masks.xy[0])
            # Calculate the minimum angle correction for bending
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
        if cable_type in [CableType.YELLOWT1, CableType.YELLOWT2, CableType.YELLOW2T1, CableType.YELLOW2T2]:
            result = self.yellow_models['kp'](image, verbose=False)[0].cpu()
        else:
            raise ValueError(f"Invalid cable type: {cable_type}")

        # Extract key points from the result
        kps = result.keypoints
        if len(kps) > 0:
            keypoints = kps.xy.numpy()[0]

            # Identify left and right pivot points
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
            raise ValueError("No cable detected in the image")

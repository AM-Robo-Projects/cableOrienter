#!/usr/bin/env python3
"""
YOOO Demo: You Only Orient Once - Vision-Driven Robot Cable Orientation Detection

This demo implements the core concepts of the YOOO approach:
1. Keypoint detection for pitch correction
2. Segmentation for roll correction
3. Visualization of the detected angles
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import sys
import math
from CableOrienter import CableOrienter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOOO Demo: Cable Orientation Detection')
    parser.add_argument('--kp_model', type=str, default='./models/yellow_kp.pt',
                        help='Path to the keypoint detection model')
    parser.add_argument('--seg_model', type=str, default='./models/yellow_seg.pt',
                        help='Path to the segmentation model')
    parser.add_argument('--pitch_image', type=str, default=None,
                        help='Path to the front view image for pitch detection')
    parser.add_argument('--roll_image', type=str, default=None,
                        help='Path to the side view image for roll detection')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualization results (if not specified, results are not saved)')
    parser.add_argument('--roll_angles', type=int, default=8,
                        help='Number of roll angle simulations to average (higher values improve stability)')
    parser.add_argument('--display', action='store_true', default=False,
                        help='Display results using matplotlib (default: False, include flag to show)')
    parser.add_argument('--display_size', type=int, default=800,
                        help='Maximum width/height for displayed images (default: 800px)')
    return parser.parse_args()


def resize_image(image, target_size=800, use_width=True):
    """Resizes an image while maintaining aspect ratio."""
    h, w = image.shape[:2]
    
    if (use_width and w <= target_size) or (not use_width and h <= target_size):
        return image.copy()
    
    if use_width:
        new_w = target_size
        new_h = int(h * (target_size / float(w)))
    else:
        new_h = target_size
        new_w = int(w * (target_size / float(h)))
        
    return cv2.resize(image, (new_w, new_h))


def draw_detection_visualization(image, detection_type, detection_data):
    """Creates a basic visualization of detection results."""
    img_copy = image.copy()
    
    if detection_type == 'kp' and 'keypoints' in detection_data:
        keypoints = detection_data['keypoints']
        
        for i, (x, y) in enumerate(keypoints):
            cv2.circle(img_copy, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(img_copy, str(i), (int(x) + 5, int(y) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(img_copy, str(i), (int(x) + 5, int(y) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        x_left, y_left = keypoints[3]
        x_right, y_right = keypoints[2]
        cv2.line(img_copy, (int(x_left), int(y_left)), (int(x_right), int(y_right)), 
                 (255, 0, 0), 4)
                
    elif detection_type == 'seg':
        if 'mask' in detection_data:
            mask = detection_data['mask']
            
            # Draw segmentation mask contour
            cv2.polylines(img_copy, [np.array(mask, dtype=np.int32)], 
                        isClosed=True, color=(0, 255, 0), thickness=4)
            
            if 'corners' in detection_data:
                # Draw approximated polygon
                corners = detection_data['corners']
                corners_array = np.array([corners[0], corners[1], corners[3], corners[2], corners[0]], dtype=np.int32)
                cv2.polylines(img_copy, [corners_array], isClosed=True, color=(255, 0, 0), thickness=6)  # Blue polygon
    
    return img_copy


def add_visible_text(image, text, position, font_size=0.9, thickness=3):
    """Adds text to an image with enhanced visibility using a black background and white text."""
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)
    
    x, y = position
    cv2.rectangle(image, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), (0, 0, 0), -1)
    
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 255),
        thickness
    )
    
    return image


def main():
    args = parse_args()
    
    # Check if at least one image is provided
    if args.pitch_image is None and args.roll_image is None:
        print("Error: At least one image must be provided using --pitch_image or --roll_image")
        print("Example: python YOOODemo.py --pitch_image ./demo_pics/kp0.jpg --roll_image ./demo_pics/seg0.jpg")
        return
    
    # Create output directory if specified and doesn't exist
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            print(f"Created output directory: {args.output_dir}")
    
    # Initialize YOOO system
    print("Initializing YOOO system...")
    start_time = time.time()
    cable_orienter = CableOrienter(args.kp_model, args.seg_model)
    print(f"Initialization complete! ({time.time() - start_time:.2f}s)")
    
    # Initialize variables to store visualization images
    pitch_viz_image = None
    pitch_angle_deg = None
    roll_viz_image = None
    roll_angle_deg = None
    
    if args.pitch_image is not None:
        if not os.path.exists(args.pitch_image):
            print(f"Error: Pitch image not found at {args.pitch_image}")
        else:
            print(f"\nProcessing pitch detection using image: {args.pitch_image}")
            pitch_image = cv2.imread(args.pitch_image)
            if pitch_image is None:
                print(f"Error: Could not read image at {args.pitch_image}")
            else:
                try:
                    start_time = time.time()
                    pitch_angle_rad = cable_orienter.determine_pitch_angle(pitch_image)
                    pitch_angle_deg = np.rad2deg(pitch_angle_rad)
                    processing_time = time.time() - start_time
                    
                    print(f"Detected pitch angle: {pitch_angle_deg:.2f}° (correction needed)")
                    print(f"Processing time: {processing_time:.3f} seconds")
                    
                    detection_data = cable_orienter.get_detection_data(pitch_image, 'kp')
                    pitch_viz_image = draw_detection_visualization(pitch_image, 'kp', detection_data)
                    
                except Exception as e:
                    print(f"Error during pitch detection: {str(e)}")
                    pitch_angle_deg = None
                    pitch_viz_image = None
    
    if args.roll_image is not None:
        if not os.path.exists(args.roll_image):
            print(f"Error: Roll image not found at {args.roll_image}")
        else:
            print(f"\nProcessing roll detection using image: {args.roll_image}")
            roll_image = cv2.imread(args.roll_image)
            if roll_image is None:
                print(f"Error: Could not read image at {args.roll_image}")
            else:
                try:
                    print(f"Simulating {args.roll_angles} roll angle measurements for weighted averaging...")
                    roll_angles = []
                    for i in range(args.roll_angles):
                        img = roll_image.copy()
                        if i > 0:
                            brightness = np.random.uniform(0.95, 1.05)
                            img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
                        
                        roll_angle_rad = cable_orienter.determine_roll_angle(img)
                        roll_angles.append(np.rad2deg(roll_angle_rad))
                    
                    weighted_angle_rad = cable_orienter.calculate_weighted_roll_angles(roll_angles)
                    roll_angle_deg = np.rad2deg(weighted_angle_rad)
                    
                    print(f"Individual roll angles: {[f'{angle:.2f}°' for angle in roll_angles]}")
                    print(f"Weighted average roll angle: {roll_angle_deg:.2f}° (correction needed)")
                    
                    detection_data = cable_orienter.get_detection_data(roll_image, 'seg')
                    roll_viz_image = draw_detection_visualization(roll_image, 'seg', detection_data)
                    
                except Exception as e:
                    print(f"Error during roll detection: {str(e)}")
                    roll_angle_deg = None
                    roll_viz_image = None
    
    # First, process and save images if output directory is specified
    if args.output_dir and (pitch_viz_image is not None or roll_viz_image is not None):
        if pitch_viz_image is not None:
            display_img_pitch = resize_image(pitch_viz_image, args.display_size)
            
            if pitch_angle_deg is not None:
                display_img_pitch = add_visible_text(
                    display_img_pitch,
                    f"Pitch Correction: {pitch_angle_deg:.2f} deg",
                    (20, 50),
                    font_size=0.6,
                    thickness=2
                )
                
            output_path = os.path.join(args.output_dir, 'pitch_detection.jpg')
            cv2.imwrite(output_path, display_img_pitch)
            print(f"Saved pitch visualization to {output_path}")
            
        if roll_viz_image is not None:
            display_img_roll = resize_image(roll_viz_image, args.display_size)
            
            if roll_angle_deg is not None:
                display_img_roll = add_visible_text(
                    display_img_roll,
                    f"Roll Correction: {roll_angle_deg:.2f} deg",
                    (20, 50),
                    font_size=0.6,
                    thickness=2
                )
                
            output_path = os.path.join(args.output_dir, 'roll_detection.jpg')
            cv2.imwrite(output_path, display_img_roll)
            print(f"Saved roll visualization to {output_path}")
    
    # Then, display images if --display flag is used
    if args.display and (pitch_viz_image is not None or roll_viz_image is not None):
        both_images = pitch_viz_image is not None and roll_viz_image is not None
        
        if both_images:
            plt.figure(figsize=(16, 8), facecolor='#333333')
        else:
            plt.figure(figsize=(9, 8), facecolor='#333333')
        
        if pitch_viz_image is not None:
            display_img_pitch = resize_image(pitch_viz_image, args.display_size)
            
            if pitch_angle_deg is not None:
                display_img_pitch = add_visible_text(
                    display_img_pitch,
                    f"Pitch Correction: {pitch_angle_deg:.2f} deg",
                    (20, 50),
                    font_size=0.6,
                    thickness=2
                )
            
            # Use different subplot configuration based on number of images
            if both_images:
                plt.subplot(1, 2, 1)
            else:
                plt.subplot(1, 1, 1)  # Center the image when it's the only one
                
            plt.imshow(cv2.cvtColor(display_img_pitch, cv2.COLOR_BGR2RGB))
            plt.title("Pitch Detection", color='white')
            plt.axis('off')
            
        if roll_viz_image is not None:
            display_img_roll = resize_image(roll_viz_image, args.display_size)
            
            if roll_angle_deg is not None:
                display_img_roll = add_visible_text(
                    display_img_roll,
                    f"Roll Correction: {roll_angle_deg:.2f} deg",
                    (20, 50),
                    font_size=0.6,
                    thickness=2
                )
            
            # Use different subplot configuration based on number of images
            if both_images:
                plt.subplot(1, 2, 2)
            else:
                plt.subplot(1, 1, 1)  # Center the image when it's the only one
                
            plt.imshow(cv2.cvtColor(display_img_roll, cv2.COLOR_BGR2RGB))
            plt.title("Roll Detection", color='white')
            plt.axis('off')
            
        plt.tight_layout()
        print("\nDisplaying detection results. Close the window to continue...")
        plt.show()
    
    print("\nYOOO Demo completed!")


if __name__ == "__main__":
    main()

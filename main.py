import cv2
import numpy as np
from CableOrienter import CableOrienter, CableType, get_approx_polly


def resize_with_aspect_ratio(image, width=None, height=None):
    """
    Resize an image while maintaining the aspect ratio.

    Parameters:
    - image: The input image
    - width: Target width (if None, will be calculated from height)
    - height: Target height (if None, will be calculated from width)

    Returns:
    - Resized image
    """
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        # Calculate the ratio based on height
        ratio = height / float(h)
        new_width = int(w * ratio)
        new_size = (new_width, height)
    else:
        # Calculate the ratio based on width
        ratio = width / float(w)
        new_height = int(h * ratio)
        new_size = (width, new_height)

    return cv2.resize(image, new_size)


def visualize_bending_detection(image, result, bending_angle_min, bending_angle_max,):
    """
    Creates visualization of bending angle detection.
    
    Parameters:
    - image: Original input image
    - result: Segmentation result
    - bending_angle_min: Minimum bending angle in radians
    - bending_angle_max: Maximum bending angle in radians
    - cable_corners: Corner points of the cable shape
    
    Returns:
    - Annotated image showing the detection
    """
    # Create a copy of the image for visualization
    vis_img = image.copy()
    
    # Create a semi-transparent overlay for the mask
    if result.masks is not None:
        xy_seg = result.masks.xy[0]
        # Convert to integer points array with the right shape for fillPoly
        points = np.array(xy_seg, dtype=np.int32).reshape((-1, 1, 2))
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillPoly(mask, [points], (0, 255, 0))
        overlay = cv2.addWeighted(vis_img, 0.5, mask, 0.5, 0)
        vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)

    top_left, top_right, bottom_left, bottom_right = get_approx_polly(result.masks.xy[0])
    cable_corners = [top_left, top_right, bottom_right, bottom_left, top_left]
    # Draw lines connecting the corners
    corners_array = np.array(cable_corners, dtype=np.int32)
    cv2.polylines(vis_img, [corners_array], True, (0, 0, 255), 10)
    
    # Add dots at each corner and a number
    for corner in cable_corners:
        cv2.circle(vis_img, (int(corner[0]), int(corner[1])), 5, (255, 0, 0), -1)
        
    # Add angle text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_img, f"Min angle: {np.rad2deg(bending_angle_min):.2f}°", 
               (10, 30), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(vis_img, f"Max angle: {np.rad2deg(bending_angle_max):.2f}°", 
               (10, 60), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
               
    return vis_img


def visualize_pitch_detection(image, result, pitch_angle):
    """
    Creates visualization of pitch angle detection.
    
    Parameters:
    - image: Original input image
    - result: Keypoint detection result
    - pitch_angle: Detected pitch angle in radians
    
    Returns:
    - Annotated image showing the detection
    """
    # Create a copy of the image for visualization
    vis_img = image.copy()
    
    # Extract keypoints
    keypoints = result.keypoints.xy.numpy()[0]
    
    # Draw all keypoints
    for i, (x, y) in enumerate(keypoints):
        cv2.circle(vis_img, (int(x), int(y)), 5, (0, 255, 255), -1)
        cv2.putText(vis_img, f"KP{i}", (int(x) + 5, int(y) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Highlight the specific keypoints used for angle calculation
    x_left, y_left = keypoints[3]
    x_right, y_right = keypoints[2]
    
    # Draw a line between the key points
    cv2.line(vis_img, (int(x_left), int(y_left)), (int(x_right), int(y_right)), 
            (255, 0, 0), 2)
    
    # Add angle text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_img, f"Pitch angle: {np.rad2deg(pitch_angle):.2f}°", 
               (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    return vis_img


def main():
    # Initialize the CableOrienter
    print("Initializing Cable Orienter system...")
    cable_orienter = CableOrienter(show_progress=True)
    print("Initialization complete!")

    # Load image paths for bending and pitch detection
    image_bending_path = "demo_pics/seg5.jpg"
    image_pitch_path = "demo_pics/kp5.jpg"

    print(f"Loading images from {image_bending_path} and {image_pitch_path}...")
    
    # Read the images
    image_bending = cv2.imread(image_bending_path)
    image_pitch = cv2.imread(image_pitch_path)

    if image_bending is None or image_pitch is None:
        print("Error: One or both of the images were not found!")
        print(f"Please ensure that the demo images exist at:")
        print(f"  - {image_bending_path}")
        print(f"  - {image_pitch_path}")
        return

    # Set cable type for detection
    cable_type = CableType.YELLOW
    print(f"Using cable type: {cable_type.name}")

    # Perform bending detection
    print("\nPerforming bending detection...")
    try:
        # Get raw results for visualization
        seg_result = cable_orienter.yellow_models['seg'](image_bending, verbose=False)[0].cpu()
        
        if seg_result.masks is None:
            print("No cable detected in the bending image!")
            return
        # Calculate angles
        bending_angle_min = cable_orienter.determine_bending(image_bending, cable_type)
        bending_angle_max = cable_orienter.determine_bending(image_bending, cable_type, use_min=False)

        # Create visualization
        bending_vis = visualize_bending_detection(image_bending, seg_result, bending_angle_min, bending_angle_max)
    except Exception as e:
        print(f"Error during bending detection: {str(e)}")
        return

    # Perform pitch detection
    print("Performing pitch detection...")
    try:
        # Get raw results for visualization
        kp_result = cable_orienter.yellow_models['kp'](image_pitch, verbose=False)[0].cpu()
        
        if len(kp_result.keypoints) == 0:
            print("No keypoints detected in the pitch image!")
            return
            
        # Calculate pitch
        pitch_angle = cable_orienter.determine_pitch(image_pitch, cable_type)
        
        # Create visualization
        pitch_vis = visualize_pitch_detection(
            image_pitch, kp_result, pitch_angle)
    except Exception as e:
        print(f"Error during pitch detection: {str(e)}")
        return

    # Print the angles
    print("\nResults:")
    print(f"Bending angle (min): {bending_angle_min:.4f} radians ({np.rad2deg(bending_angle_min):.2f}°)")
    print(f"Bending angle (max): {bending_angle_max:.4f} radians ({np.rad2deg(bending_angle_max):.2f}°)")
    print(f"Pitch angle: {pitch_angle:.4f} radians ({np.rad2deg(pitch_angle):.2f}°)")

    # Resize images for display
    print("\nPreparing visualization...")
    resized_image_bending = resize_with_aspect_ratio(bending_vis, width=800)
    resized_image_pitch = resize_with_aspect_ratio(pitch_vis, width=800)

    # Display the images
    print("Displaying results. Press any key to exit.")
    cv2.imshow("Bending Angle Detection", resized_image_bending)
    cv2.imshow("Pitch Angle Detection", resized_image_pitch)

    # Wait until a key is pressed to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Demo completed successfully!")


# Example of how to use this in other applications
def example_usage():
    """
    This function demonstrates how to integrate Cable Orienter into your own applications.
    """
    from CableOrienter import CableOrienter, CableType
    import cv2
    
    # Initialize the system
    cable_orienter = CableOrienter()
    
    # Load your image
    image = cv2.imread("your_cable_image.jpg")
    
    # Select the appropriate cable type
    cable_type = CableType.YELLOW
    
    # Detect bending and pitch angles
    bending_angle = cable_orienter.determine_bending(image, cable_type)
    pitch_angle = cable_orienter.determine_pitch(image, cable_type)
    
    # Use the angles in your application
    print(f"Bending angle: {np.rad2deg(bending_angle):.2f} degrees")
    print(f"Pitch angle: {np.rad2deg(pitch_angle):.2f} degrees")


if __name__ == "__main__":
    main()

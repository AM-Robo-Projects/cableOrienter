import cv2
from CableOrienter import CableOrienter, CableType


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


def main():
    # Initialize the CableOrienter
    cable_orienter = CableOrienter()

    # Load image paths for bending and pitch detection
    image_bending_path = "demo_pics/seg5.jpg"
    image_pitch_path = "demo_pics/kp5.jpg"

    # Read the images
    image_bending = cv2.imread(image_bending_path)
    image_pitch = cv2.imread(image_pitch_path)

    if image_bending is None or image_pitch is None:
        raise ValueError("One or both of the images were not found!")

    # Perform bending and pitch detection
    pitch_angle = cable_orienter.determine_pitch(image_pitch, CableType.YELLOWT1)
    bending_angle_min = cable_orienter.determine_bending(image_bending, CableType.YELLOWT1)
    bending_angle_max = cable_orienter.determine_bending(image_bending, CableType.YELLOWT1, use_min=False)

    # Print the angles
    print(f"Bending angle (min): {bending_angle_min} radians")
    print(f"Bending angle (max): {bending_angle_max} radians")
    print(f"Pitch angle: {pitch_angle} radians")

    resized_image_bending = resize_with_aspect_ratio(image_bending, width=800)
    resized_image_pitch = resize_with_aspect_ratio(image_pitch, width=800)

    # Overlay the angles on the resized images
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_bending_min = f"Bending angle (min): {bending_angle_min:.2f} radians"
    text_bending_max = f"Bending angle (max): {bending_angle_max:.2f} radians"
    text_pitch = f"Pitch angle: {pitch_angle:.2f} radians"

    # Display bending image with min bending angle and max bending angle
    cv2.putText(resized_image_bending, text_bending_min, (10, 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(resized_image_bending, text_bending_max, (10, 80), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Bending Angle", resized_image_bending)

    # Display pitch image with pitch angle
    cv2.putText(resized_image_pitch, text_pitch, (10, 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Pitch Angle", resized_image_pitch)

    # Wait until a key is pressed to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

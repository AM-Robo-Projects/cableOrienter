# YOOO: You Only Orient Once - Cable Orientation Demo

A vision-driven approach for precise cable orientation detection using YOLOv11 models.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the demo with image paths:

```bash
python YOOODemo.py --pitch_image ./demo_pics/kp0.jpg --roll_image ./demo_pics/seg0.jpg --display
```

Save results to output directory:

```bash
python YOOODemo.py --roll_image ./new_pics/seg_new1.jpg --output_dir output
```

## Key Command Line Options

- `--pitch_image`: Front view image for pitch angle detection
- `--roll_image`: Side view image for roll angle detection
- `--output_dir`: Directory to save visualization results
- `--display`: Enable visual display of results (flag with no value)
- `--roll_angles`: Number of roll angle simulations to average (default: 8)

## CableOrienter API Example

```python
from CableOrienter import CableOrienter
import cv2

# Initialize
orienter = CableOrienter(
    keypoint_model_path='./models/yellow_kp.pt',
    segmentation_model_path='./models/yellow_seg.pt'
)

# Load images
pitch_image = cv2.imread('path/to/pitch_image.jpg')
roll_image = cv2.imread('path/to/roll_image.jpg')

# Get angles
pitch_angle = orienter.determine_pitch_angle(pitch_image)  # in radians
roll_angle = orienter.determine_roll_angle(roll_image)     # in radians

# Get detection data for visualization
pitch_data = orienter.get_detection_data(pitch_image, 'kp')
roll_data = orienter.get_detection_data(roll_image, 'seg')
```

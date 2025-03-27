# Cable Orienter Project

A computer vision system for detecting cable orientation parameters (pitch and bending) using YOLO models.

## Project Overview

This project provides a utility for measuring cable orientation to assist robotic manipulation tasks. It uses:

- YOLO models for keypoint detection and segmentation
- Computer vision techniques for angle calculation
- Support for various cable types

## Setup & Installation

1. Clone this repository:
   ```
   git clone https://github.com/AM-Robo-Projects/WiringHarnessChallenge_2.0.git
   cd WiringHarnessChallenge_2.0
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script to see a demonstration:

```
python main.py
```

For integrating into your own project:

```python
from CableOrienter import CableOrienter, CableType

# Initialize the orienter
cable_orienter = CableOrienter()

# Detect bending angle
bending_angle = cable_orienter.determine_bending(image, CableType.YELLOW)

# Detect pitch angle
pitch_angle = cable_orienter.determine_pitch(image, CableType.YELLOW)
```

## Project Website

Visit our [project website](https://AM-Robo-Projects.github.io/WiringHarnessChallenge_2.0/) for additional documentation and examples.

## Citation

If you use this project in your research, please cite:

```
@article{cableorienter2024,
    title={Cable Orienter: Precise Cable Pitch and Bending Detection for Robotic Applications},
    author={AM-Robo-Projects},
    journal={arXiv preprint arXiv:XXXX.XXXXX},
    year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

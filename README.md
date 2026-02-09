# Study Hall Seat Detection System 

**Course:** Image Processing and Computer Vision  
**Institution:** Politecnico di Torino  
**Academic Year:** 2021/2022  

## Overview
Finding a free seat in university study halls can be time-consuming. This project proposes a Computer Vision solution to automatically detect and visualize available seats in a study hall environment.

The system analyzes video footage to identify students and map their positions against pre-defined seat coordinates. To ensure stability and accuracy, it combines deep learning object detection (**YOLOv4**) with a custom logic algorithm that handles occlusion and temporary movements.

## Project Structure
Based on the repository organization:

```text
├── adaptive_th/      # Experiments with Adaptive Thresholding
├── cv_dnn/           # OpenCV DNN module tests
├── saving_frame/     # Utilities for frame extraction
├── yolo_dnn/         # YOLO implementation files
├── main.py           # CORE: Main script for logic and visualization
├── Motion_est.py     # Experiments with Motion Estimation (Optical Flow)
├── SpacePicker.py    # CORE: Tool to manually define seat coordinates
├── VideoPickle       # Serialized file containing seat coordinates
├── result.json       # Pre-computed YOLOv4 detections for the demo video
├── Report.pdf        # Detailed project documentation (Italian)
└── README.md         # Project documentation
```
## Methodology
The final solution is implemented in a two-stage pipeline:

1. Detection (YOLOv4): * We utilized the YOLOv4 neural network (trained on the COCO dataset) to detect persons in the video feed.

- Due to hardware constraints, detection was performed on Google Colab, and the bounding box results were exported to result.json.

- *Note*: The system runs at a reduced framerate (approx. 16 fps) to optimize processing.

2. Logic & Visualization (main.py):

- Mapping: The script loads the pre-defined seat areas (ROI) from VideoPickle.
- Occupancy Check: It checks if the center of a detected person falls within a seat's ROI.
- Temporal Robustness: To prevent flickering (e.g., a student leaning out of frame), a counter system is used. A seat is marked "Occupied" only after a person is detected for FRAME_ROBUSTNESS consecutive frames. Conversely, it only switches to "Free" after being empty for a specific threshold.
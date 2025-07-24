# FramePoint

FramePoint is a browser-based person tracking system using TensorFlow.js. It combines COCO-SSD and MoveNet to detect and track people in real time. The camera feed is rendered on a fixed 4K canvas, with logic for zooming and centering based on eye position.

## Core Functionality

- Uses COCO-SSD for object detection (optional, currently not active in logic)
- Uses MoveNet (SINGLEPOSE_LIGHTNING) for pose detection
- Tracks the midpoint between left and right eye as the focus point
- Dynamically adjusts the position of the visible frame to keep the face centered
- Estimates depth (z) from bounding box size of keypoints (eyes, nose, ears)
- Displays real-time data: x/y/z coordinates, FPS, detection time, CPU usage

## How It Works

1. `index.html` loads the TensorFlow models and sets up video + canvas
2. `script.js`:
   - Initializes video stream and canvas (3840×2160)
   - Detects pose keypoints via MoveNet
   - Calculates center point (x/y) and approximate z-distance
   - Draws the video frame and overlays tracking visuals
3. The green dot represents the computed center of the detected face
4. Stats are updated periodically and shown in the top-left corner

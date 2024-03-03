# Pedestrian Detection and Pose Estimation

This project utilizes computer vision and machine learning techniques to detect pedestrians in a video stream, count them, and estimate their poses. The implementation is based on Python using OpenCV, Mediapipe, and the HOG (Histogram of Oriented Gradients) algorithm.

## Overview

The main objectives of this project are:

1. Detect pedestrians in a video stream using the HOG algorithm.
2. Count the number of pedestrians detected.
3. Estimate the pose of each pedestrian using Mediapipe.
4. Visualize the detected pedestrians and their estimated poses in real-time.

## Dependencies

- Python
- OpenCV
- Imutils
- Mediapipe

## Usage

1. Clone the repository or download the Python script.
2. Install the required dependencies.
3. Run the Python script.
4. Provide the path to the input video file when prompted.
5. The script will process the video, detecting pedestrians, estimating their poses, and displaying the results in real-time.
6. Press 'q' to exit the application.

## Code Structure

The Python script `pedestrian_detection_pose_estimation.py` contains the implementation. It follows these main steps:

1. Initialize OpenCV, Mediapipe, and the HOG algorithm.
2. Open the input video file and initialize the output video writer.
3. Read frames from the video stream.
4. Detect pedestrians using the HOG algorithm.
5. Count the number of detected pedestrians and draw bounding boxes around them.
6. Estimate the pose of each pedestrian using Mediapipe.
7. Visualize the detected pedestrians and their estimated poses.
8. Write the processed frames to the output video file.
9. Exit the application when the video processing is complete.

## Future Enhancements

- Fine-tuning the pedestrian detection algorithm for improved accuracy.
- Enhancing the pose estimation model to handle occlusions and complex poses.
- Integrating with additional sensors or cameras for better environmental perception.

## Contributions

Contributions to this project are welcome. Feel free to fork the repository, make changes, and submit pull requests.

---

This repository contains the code for pedestrian detection and pose estimation using computer vision techniques. Let's make our streets safer with intelligent pedestrian monitoring!

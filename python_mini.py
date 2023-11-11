# -*- coding: utf-8 -*-
"""Python_mini.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AVkZg7TQS3Yjdb7McxZs6JI6b_jl3JeY
"""

from google.colab import drive
drive.mount('/content/drive')

import cv2
import imutils
import mediapipe as mp
from google.colab.patches import cv2_imshow

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture('/content/drive/MyDrive/vid.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/content/drive/MyDrive/output1.mp4', fourcc, 25.0, (640, 480))

while cap.isOpened():
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=min(400, image.shape[1]))
        (regions, _) = hog.detectMultiScale(image,
                                             winStride=(4, 4),
                                             padding=(4, 4),
                                             scale=1.05)
        count = 0
        for (x, y, w, h) in regions:
            count += 1
            pedestrian_roi = image[y:y+h, x:x+w]
            RGB = cv2.cvtColor(pedestrian_roi, cv2.COLOR_BGR2RGB)
            results = pose.process(RGB)
            mp_drawing.draw_landmarks(
                pedestrian_roi, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            image[y:y+h, x:x+w] = pedestrian_roi


        cv2.putText(image, f"Pedestrians: {count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(image)
        cv2_imshow(image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

"""
import cv2
import imutils
import mediapipe as mp
from google.colab.patches import cv2_imshow

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture('/content/drive/MyDrive/vid.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/content/drive/MyDrive/output1.mp4', fourcc, 25.0, (640, 480))

while cap.isOpened():
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=min(400, image.shape[1]))
        (regions, _) = hog.detectMultiScale(image,
                                             winStride=(4, 4),
                                             padding=(4, 4),
                                             scale=1.05)
        count = 0
        for (x, y, w, h) in regions:
            count += 1
            pedestrian_roi = image[y:y+h, x:x+w]
            RGB = cv2.cvtColor(pedestrian_roi, cv2.COLOR_BGR2RGB)
            results = pose.process(RGB)
            mp_drawing.draw_landmarks(
                pedestrian_roi, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            image[y:y+h, x:x+w] = pedestrian_roi


        cv2.putText(image, f"Pedestrians: {count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(image)
        cv2_imshow(image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows("""

pip install mediapipe
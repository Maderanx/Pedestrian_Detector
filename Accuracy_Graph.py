import cv2 as cv
import imutils
import mediapipe as mp
import numpy
import os
import matplotlib.pyplot as plt

# number of actual pedestrians
pedestrians = 0

# OpenCV objects/instances
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# histogram of oriented gradients
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector()) # support vector machine

# post detection
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

x = numpy.linspace(1, 20549 - 40 + 1, (20549 - 40 + 1) - 1 + 1)
y1 = list()

# iterate through all images in dataset

start = 40
end = 500

for i in range(start, end + 1):
    print(i)
    image_path = './WiderPerson/Images/' + (str(i)).zfill(6) + '.jpg'
    if os.path.exists(image_path):
        image = cv.imread(image_path)
        image = imutils.resize(image, width=min(400, image.shape[1]))

        (regions, _) = hog.detectMultiScale(image,
                                                winStride=(4, 4),
                                                padding=(4, 4),
                                                scale=1.05)
        
        count = 0  
        for (x, y, w, h) in regions:
            count += 1  
            pedestrian_roi = image[y:y+h, x:x+w]
            RGB = cv.cvtColor(pedestrian_roi, cv.COLOR_BGR2RGB)
            results = pose.process(RGB)
            mp_drawing.draw_landmarks(pedestrian_roi, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            image[y:y+h, x:x+w] = pedestrian_roi
        
        # compare count with dataset's number of pedestrians
        file_address = './WiderPerson/Annotations/' + (str(i)).zfill(6) + '.jpg.txt'

        if os.path.exists(file_address):
            with open(file_address) as f:
                pedestrians = int(f.readline())
                # calculate cost function
                cost_function = (count - pedestrians)**2
                # maintain array of x and y
                y1.append(cost_function)

        # # find the pedestrian count from file_address
        # f = open(file_address, 'r')
        # if f:
        #     pedestrians = int(f.readline())
        #     # calculate cost function
        #     cost_function = (count - pedestrians)**2
        #     # maintain array of x and y
        #     y1.append(cost_function)
        # else:
        #     y1.append(0)

y = numpy.array(y1)
x = numpy.array([i for i in range(len(y1))])
print(y)
plt.plot(x, y)
plt.xlabel('Iteration')
plt.ylabel('Loss function')
plt.show()
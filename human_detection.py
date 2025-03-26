#human detection using various methods

import cv2
import mediapipe as mp

#initialize mediapipe pose
mp_pose = mp.solutions.pose
mp.drawing = mp.solutions.drawing_utils

#create pose object which contains the model
pose = mp_pose.Pose()

#open webcam for input
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break

    #convert RGB to BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #process frame with mediapipe neural network,returns landmark positions
    results = pose.process(rgb_frame)

    #draw skeleton on frame
    if results.pose_landmarks:
        mp.drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    #show output
    cv2.imshow('skeleton drawing',frame)

    #exit 
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break


#release everything
cap.release()
cv2.destroyAllWindows()


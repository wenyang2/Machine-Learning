#human detection using various methods

import cv2
import mediapipe as mp
from filterpy.kalman import KalmanFilter 
import numpy as np

#initialize mediapipe pose
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

#create pose object which contains the model,provide landmark smoothing
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.2)

#create Kalman filter function
def kalman_filter():
    #initialize 1st order or 2nd order
    kf=KalmanFilter(dim_x=4, dim_z=2)
    #state transition matrix
    kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
    #measurement matrix
    kf.H=np.array([[1,0,0,0],[0,1,0,0]])
    #initial uncertainty
    kf.P *=1000
    #measurement noise (high value means less trust in the measurement, in this case the mediapipe landmark positions)
    kf.R *= 0.8
    #process noise (uncertainty in the model of motion, dynamic and unpredictable, or smooth and prdictable)
    #high value means kf reacts faster to changes in measurement
    kf.Q *= 10.0

    return kf
#create kf object for each landmark (there are 33 landmarks for mediapipe pose)
kalman_filters =  {i:kalman_filter() for i in range(33)}

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
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        #run land mark positions through the kalman filters
        for i, j in enumerate(results.pose_landmarks.landmark):
            #get the x,y coordinates of the landmark
            #since landmark is normalized to [0,1], we need to multiply by frame width and height to get pixel coordinates
            x = int(j.x * frame.shape[1])
            y = int(j.y * frame.shape[0])

            #update the kalman filter with the new measurement
            kalman_filters[i].update(np.array([x, y]))
            kalman_filters[i].predict()

            #get the predicted position from the kalman filter (first two elements from state vector in each landmark)
            pred_x, pred_y = kalman_filters[i].x[:2]

            #draw the predicted position on the frame, BGR format for cv2
            cv2.circle(frame, (int(pred_x), int(pred_y)), 5, (0, 255, 0), -1)

    #show output
    cv2.imshow('skeleton drawing',frame)

    #exit 
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break


#release everything
cap.release()
cv2.destroyAllWindows()
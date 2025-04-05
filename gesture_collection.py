import cv2
import mediapipe as mp
import os
import csv

#extracts hand landmarks from video via mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

#output file to save hand landmarks into
output_file="hand_landmarks.csv"

#if file doesn't exist, write a new file 
if not os.path.exists(output_file):
    with open(output_file,mode='w',newline='') as file:
        writer = csv.writer(file)
        header=['label']+[f'{i}_{axis}' for i in range(21) for axis in ['x','y','z']]
        writer.writerow(header)

#setup the gesture label mappings
gesture_labels={
    ord('p'):'peace_sign',
    ord('f'):'fist',
    ord('m'):'middle_finger'
}

#capture webcam video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break

    #flip the output (like mirror for more intuitive use)
    frame=cv2.flip(frame,1)

    #convert to rgb
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    #process with mediapipe model to get hand landmarks
    results=hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #draw the hand landmarks on the frame
            mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

            #press a key to record the current hand landmarks
            key=cv2.waitKey(10)
            if key!=-1 and key!= ord('q'):
                #check if key exists in gesture_labels
                if key in gesture_labels:
                    label=gesture_labels[key]

                    #initialize empty list for storing current landmark data
                    landmark_data=[]
                    #append the current landmarks into list
                    for lm in hand_landmarks.landmark:
                        landmark_data.extend([lm.x,lm.y,lm.z])

                    #append into the csv file
                    with open(output_file, mode='a',newline='')as f:
                        writer=csv.writer(f)
                        writer.writerow([label]+landmark_data)
                    
                    print(f"succesfully saved gesture {label}")

    #output
    cv2.imshow('hand landmarks',frame)
    #exit
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break

#release everything
cap.release()
cv2.destroyAllWindows()

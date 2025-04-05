import cv2
import mediapipe as mp

#extracts hand landmarks from video via mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils


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
    
    #draw the hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
    #output
    cv2.imshow('hand landmarks',frame)
    #exit
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break

#release everything
cap.release()
cv2.destroyAllWindows()

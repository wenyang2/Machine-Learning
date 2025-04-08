import cv2 
import mediapipe as mp

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw=mp.solutions.drawing_utils

#open webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break

    frame=cv2.flip(frame,1)
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #process the frame with mediapipe hand model
    results = hands.process(rgb_frame)

    #if hand is detected in the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)



    cv2.imshow('gesture detection',frame)
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
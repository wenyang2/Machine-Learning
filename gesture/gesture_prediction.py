import cv2 
import mediapipe as mp
from gesture_models import mlp_classifier
import torch 
import json

#load mediapipe models for hand landmark labeling
mp_hands=mp.solutions.hands
hands=mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw=mp.solutions.drawing_utils

#load our own model for classification
model = mlp_classifier(num_classes=5) #5 classes of prediction
model.load_state_dict(torch.load('.\gesture_mlp.pt')) #load our model
model.eval() #set to inference mode /evaluation mode

#get our reverse label map
with open("gesture_label_map.json","r")as f:
    label_map = json.load(f)
reverse_label_map = {a:b for b,a in label_map.items()}

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
    #if no landmarks detected
    predicted_label='no hand landmark detected.'
    #if hand is detected in the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #draw landmark on frame
            mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            #get landmark xyz value
            landmarks=[]
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x,lm.y,lm.z])
            #convert to tensor, unsqueeze turns 1D tensor into 2D tensor(adds a new dimension at index 0)
            #2D tensor shape is compulsary for pytorch models
            landmarks_tensor=torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)

        #make prediction with our model
        with torch.no_grad():
            outputs=model(landmarks_tensor)
            _,predicted = torch.max(outputs,1)
        
        #get our label name based on predicted result
        predicted_label=reverse_label_map[predicted.item()]

    print(predicted_label)
    cv2.imshow('gesture detection',frame)
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
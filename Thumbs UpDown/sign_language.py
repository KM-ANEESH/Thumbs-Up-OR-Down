import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Flip the frame horizontally for a more intuitive mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert the image to RGB and process it with MediaPipe Hands
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the landmarks for the thumb and index finger
            thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Calculate the distance between the thumb and index finger landmarks
            dist = ((thumb_landmark.x - index_landmark.x)**2 + (thumb_landmark.y - index_landmark.y)**2)**0.5
            
            # Determine if the thumb is up or down based on the distance threshold
            if dist < 0.025:
                cv2.putText(frame, "Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif thumb_landmark.y > hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y:
                cv2.putText(frame, "Thumbs Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
    cv2.imshow('Hand Gestures', frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp 

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    succ, img = cap.read()
    if not succ:
        print("Failed to capture video frame.")
        break

    img = cv2.flip(img, 1) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    
    results1 = pose.process(img_rgb)
    results2 = hands.process(img_rgb)

    if results1.pose_landmarks:
        mp_draw.draw_landmarks(img, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    
    if results2.multi_hand_landmarks:
        for hand_landmarks in results2.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    cv2.imshow("Pose and Hand Estimation", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



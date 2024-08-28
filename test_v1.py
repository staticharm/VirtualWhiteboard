import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe and OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Initialize the whiteboard
whiteboard = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Variables to track mode and hand positions
is_drawing = False
prev_x, prev_y = None, None

# Define the button positions
button_draw = (50, 50, 200, 100)  # (x1, y1, x2, y2)
button_erase = (250, 50, 400, 100)

# Function to detect if a hand is over a button
def is_hand_over_button(hand_landmarks, button):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    h, w, _ = frame_bgr.shape
    cx, cy = int(index_tip.x * w), int(index_tip.y * h)
    x1, y1, x2, y2 = button
    return x1 < cx < x2 and y1 < cy < y2



# Function to draw buttons on the frame
def draw_buttons(frame):
    cv2.rectangle(frame, (button_draw[0], button_draw[1]), (button_draw[2], button_draw[3]), (0, 255, 0), -1)
    cv2.putText(frame, "Draw", (button_draw[0] + 10, button_draw[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(frame, (button_erase[0], button_erase[1]), (button_erase[2], button_erase[3]), (0, 0, 255), -1)
    cv2.putText(frame, "Erase", (button_erase[0] + 10, button_erase[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def is_drawing_gesture(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    distance = np.sqrt((index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2 + (index_tip.z - middle_tip.z) ** 2)
    return distance < 0.1

# Define gestures for writing
# def is_writing_gesture(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    distance = np.sqrt((index_tip.x - index_mcp.x) ** 2 + (index_tip.y - index_mcp.y) ** 2 + (index_tip.z - index_mcp.z) ** 2)
    
    # Check if other fingers are folded
    folded_threshold = 0.1  # Adjust this threshold as needed
    folded = True
    for id in [mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
               mp_hands.HandLandmark.RING_FINGER_TIP, 
               mp_hands.HandLandmark.PINKY_TIP]:
        tip = hand_landmarks.landmark[id]
        base = hand_landmarks.landmark[id - 2]  # Corresponding MCP joint
        if np.sqrt((tip.x - base.x) ** 2 + (tip.y - base.y) ** 2 + (tip.z - base.z) ** 2) > folded_threshold:
            folded = False
            break
            
    return folded and distance > 0.05  # Ensure index finger is extended

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        break

    
    frame_bgr = cv2.flip(frame_bgr, 1)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if is_hand_over_button(hand_landmarks, button_draw):
                is_drawing = True
            elif is_hand_over_button(hand_landmarks, button_erase):
                is_drawing = False

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame_bgr.shape
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)

            if prev_x is not None and prev_y is not None:
                
                if is_drawing and not is_drawing_gesture(hand_landmarks):
                        cv2.line(whiteboard, (prev_x, prev_y), (cx, cy), (0, 0, 0), 5)
                elif not is_drawing:
                    cv2.line(whiteboard, (prev_x, prev_y), (cx, cy), (255, 255, 255), 20)

            prev_x, prev_y = cx, cy

    else:
        prev_x, prev_y = None, None

    # Overlay the whiteboard on the frame
    combined_image = cv2.addWeighted(frame_bgr, 0.5, whiteboard, 0.5, 0)
    
    # Draw the buttons on the frame
    draw_buttons(combined_image)
    
    # Display mode status
    mode_text = "Erasing" if not is_drawing else "Drawing"
    cv2.putText(combined_image, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the combined image
    cv2.imshow('Virtual Whiteboard', combined_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        whiteboard[:] = 255

cap.release()
cv2.destroyAllWindows()

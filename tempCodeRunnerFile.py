import cv2
import numpy as np
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def create_button(image, text, x, y, w, h, color, text_color=(0, 0, 0)):
    """Draws a filled rectangle with a border and text for UI interactions."""
    cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)


def main():
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    if not success:
        print("Error: Could not access the webcam.")
        return

    print("---------------------------------------")
    print("Virtual Air Painter (MediaPipe Tasks AI Edition)")
    print("---------------------------------------")
    
    # 1. Download the modern MediaPipe Hand Landmarker model if it doesn't exist
    model_path = 'hand_landmarker.task'
    if not os.path.exists(model_path):
        print(f"Downloading robust hand tracking model...")
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")

    # 2. Initialize the modern MediaPipe Tasks Vision API (Works fully on Python 3.13)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    detector = vision.HandLandmarker.create_from_options(options)

    canvas = np.zeros_like(frame)

    colors_dict = {
        "Blue": (255, 0, 0),
        "Green": (0, 255, 0),
        "Red": (0, 0, 255),
        "Yellow": (0, 255, 255),
        "Eraser": (0, 0, 0)
    }
    
    current_color = colors_dict["Blue"]
    brush_thickness = 5
    px, py = 0, 0
    
    button_height, button_y = 50, 10
    buttons = [
        {"name": "CLEAR", "rect": (10, button_y, 70, button_height), "bg": (200, 200, 200), "fg": (0, 0, 0)},
        {"name": "BLU",   "rect": (90, button_y, 50, button_height), "bg": colors_dict["Blue"], "fg": (255, 255, 255)},
        {"name": "GRN",   "rect": (150, button_y, 50, button_height), "bg": colors_dict["Green"], "fg": (0, 0, 0)},
        {"name": "RED",   "rect": (210, button_y, 50, button_height), "bg": colors_dict["Red"], "fg": (255, 255, 255)},
        {"name": "YLW",   "rect": (270, button_y, 50, button_height), "bg": colors_dict["Yellow"], "fg": (0, 0, 0)},
        {"name": "ERASE", "rect": (330, button_y, 70, button_height), "bg": (255, 255, 255), "fg": (0, 0, 0)},
        {"name": "THK+",  "rect": (410, button_y, 50, button_height), "bg": (150, 150, 150), "fg": (0, 0, 0)},
        {"name": "THK-",  "rect": (470, button_y, 50, button_height), "bg": (150, 150, 150), "fg": (0, 0, 0)},
        {"name": "QUIT",  "rect": (530, button_y, 60, button_height), "bg": (0, 0, 150), "fg": (255, 255, 255)},
    ]

    hover_frames = 0
    last_hover_btn = None
    HOVER_THRESHOLD = 15

    print("\nHow to use:")
    print("1. Raise your INDEX finger to draw magically in the air.")
    print("2. Raise your INDEX and MIDDLE finger completely UP to PAUSE drawing.")

    # 3. Core Loop
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        output_frame = frame.copy()
        h, w, _ = frame.shape

        # Convert to RGB and map to MediaPipe Image object format
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Detect hands using modern Tasks API
        detection_result = detector.detect(mp_image)

        cx, cy = 0, 0
        detected = False
        is_middle_up = False
        
        smoothening = 0.5
        prev_x, prev_y = 0, 0

        if hasattr(detection_result, 'hand_landmarks') and detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Landmark 8 is Index Finger Tip
                index_tip = hand_landmarks[8]
                target_x = int(index_tip.x * w)
                target_y = int(index_tip.y * h)

                cx = int(prev_x + (target_x - prev_x) * smoothening)
                cy = int(prev_y + (target_y - prev_y) * smoothening)

                prev_x, prev_y = cx, cy
                
                # Landmarks 12 and 10 are Middle Finger Tip and PIP Joint
                middle_tip = hand_landmarks[12]
                middle_pip = hand_landmarks[10]
                
                # If middle finger is physically projected distinctly higher than its knuckle
                is_middle_up = middle_tip.y < middle_pip.y - 0.02
                
                detected = True

                # Draw a nice tracking cursor on the index finger tip
                cv2.circle(output_frame, (cx, cy), 15, current_color, cv2.FILLED)
                cv2.circle(output_frame, (cx, cy), 17, (255, 255, 255), 2)
                
                # Check Interface Boundaries top of screen
                if cy <= button_y + button_height + 15:
                    px, py = 0, 0
                    current_hover_btn = None

                    for btn in buttons:
                        bx, by, bw, bh = btn["rect"]
                        if bx < cx < bx + bw and by < cy < by + bh:
                            current_hover_btn = btn["name"]
                            cv2.rectangle(output_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
                            break
                            
                    if current_hover_btn and current_hover_btn == last_hover_btn:
                        hover_frames += 1
                        cv2.putText(output_frame, f"Hold: {hover_frames}/{HOVER_THRESHOLD}", (cx - 40, cy + 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if hover_frames >= HOVER_THRESHOLD:
                            if current_hover_btn == "CLEAR": canvas = np.zeros_like(frame)
                            elif current_hover_btn == "BLU": current_color = colors_dict["Blue"]
                            elif current_hover_btn == "GRN": current_color = colors_dict["Green"]
                            elif current_hover_btn == "RED": current_color = colors_dict["Red"]
                            elif current_hover_btn == "YLW": current_color = colors_dict["Yellow"]
                            elif current_hover_btn == "ERASE": current_color = colors_dict["Eraser"]
                            elif current_hover_btn == "THK+": brush_thickness = min(40, brush_thickness + 2)
                            elif current_hover_btn == "THK-": brush_thickness = max(2, brush_thickness - 2)
                            elif current_hover_btn == "QUIT": cap.release(); cv2.destroyAllWindows(); return
                            
                            hover_frames = 0
                            last_hover_btn = None
                    else:
                        hover_frames = 0
                        last_hover_btn = current_hover_btn

                else:
                    # Canvas Drawing Actions
                    if is_middle_up:
                        px, py = 0, 0 # Act as purely movement, pause brush
                        cv2.putText(output_frame, "Pen Up (Paused)", (cx + 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        if px == 0 and py == 0:
                            px, py = cx, cy

                        if current_color == colors_dict["Eraser"]:
                            cv2.line(canvas, (px, py), (cx, cy), (0, 0, 0), brush_thickness * 4)
                        else:
                            cv2.line(canvas, (px, py), (cx, cy), current_color, brush_thickness)
                        
                        px, py = cx, cy
                        
        if not detected:
            px, py = 0, 0

        # Output blending for the Virtual Air Canvas effect
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY_INV)
        
        output_frame_bg = cv2.bitwise_and(output_frame, output_frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=cv2.bitwise_not(mask_inv))
        
        final_output = cv2.add(output_frame_bg, canvas_fg)

        # Render Top Menus
        for btn in buttons:
            bx, by, bw, bh = btn["rect"]
            create_button(final_output, btn["name"], bx, by, bw, bh, btn["bg"], btn["fg"])

        mode_text = "Eraser" if current_color == colors_dict["Eraser"] else "Brush"
        cv2.putText(final_output, f"Mode: {mode_text} | Thk: {brush_thickness} | Two Fingers UP = Pause Brush", 
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imshow("Virtual Air Painter", final_output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'): canvas = np.zeros_like(frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
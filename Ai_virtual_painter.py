import cv2
import numpy as np
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# DRAW ENGINE (IMPROVED)
# =========================
class DrawingEngine:
    def __init__(self, width, height):
        self.canvas = np.zeros((height, width, 3), np.uint8)
        self.drawColor = (0, 0, 255)
        self.thickness = 10

        self.xp, self.yp = 0, 0

        # stability
        self.smooth = 0.6
        self.prev_x, self.prev_y = 0, 0
        self.max_jump = 50  # prevents sudden jumps

    def clear(self):
        self.canvas[:] = 0
        self.xp, self.yp = 0, 0
        self.prev_x, self.prev_y = 0, 0

    def draw(self, x, y):
        # smoothing
        cx = int(self.prev_x + (x - self.prev_x) * self.smooth)
        cy = int(self.prev_y + (y - self.prev_y) * self.smooth)

        self.prev_x, self.prev_y = cx, cy

        # prevent jump
        if self.xp != 0 and self.yp != 0:
            dist = np.hypot(cx - self.xp, cy - self.yp)
            if dist > self.max_jump:
                self.xp, self.yp = cx, cy
                return

        if self.xp == 0 and self.yp == 0:
            self.xp, self.yp = cx, cy

        cv2.line(self.canvas, (self.xp, self.yp), (cx, cy),
                 self.drawColor, self.thickness)

        self.xp, self.yp = cx, cy

    def reset_cursor(self):
        self.xp, self.yp = 0, 0


# =========================
# BUTTON UI
# =========================
def draw_button(img, btn, hover=False):
    x, y, w, h = btn["rect"]
    color = btn["color"]

    # hover glow
    if hover:
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

    cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)

    cv2.putText(img, btn["name"], (x+8, y+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


# =========================
# MAIN
# =========================
def main():
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()

    if not success:
        print("Camera error")
        return

    h, w, _ = frame.shape

    engine = DrawingEngine(w, h)

    # =========================
    # MODEL SETUP
    # =========================
    model_path = "hand_landmarker.task"

    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        num_hands=1,
        min_hand_detection_confidence=0.8,
        min_hand_presence_confidence=0.8,
        min_tracking_confidence=0.8
    )

    detector = vision.HandLandmarker.create_from_options(options)

    # =========================
    # COLORS
    # =========================
    colors = {
        "BLU": (255, 0, 0),
        "GRN": (0, 255, 0),
        "RED": (0, 0, 255),
        "YLW": (0, 255, 255),
        "ERASE": (0, 0, 0)
    }

    current_color = colors["BLU"]

    # =========================
    # BUTTONS (COLORED)
    # =========================
    buttons = [
        {"name": "CLEAR", "rect": (10, 10, 90, 50), "color": (200, 200, 200)},
        {"name": "BLU", "rect": (110, 10, 60, 50), "color": colors["BLU"]},
        {"name": "GRN", "rect": (180, 10, 60, 50), "color": colors["GRN"]},
        {"name": "RED", "rect": (250, 10, 60, 50), "color": colors["RED"]},
        {"name": "YLW", "rect": (320, 10, 60, 50), "color": colors["YLW"]},
        {"name": "ERASE", "rect": (390, 10, 80, 50), "color": (255, 255, 255)},
        {"name": "QUIT", "rect": (480, 10, 80, 50), "color": (0, 0, 150)},
    ]

    running = True
    hover_frames = 0
    last_btn = None
    HOVER_THRESHOLD = 8

    fist_frames = 0
    FIST_THRESHOLD = 10

    # =========================
    # LOOP
    # =========================
    while running:

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        output = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect(mp_image)

        cx, cy = 0, 0
        middle_up = False
        closed_hand = False

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            # finger detection
            fingers = []
            fingers.append(1 if pts[4][0] < pts[3][0] else 0)

            for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
                fingers.append(1 if pts[tip][1] < pts[pip][1] else 0)

            closed_hand = all(f == 0 for f in fingers)
            middle_up = pts[12][1] < pts[10][1]

            cx, cy = pts[8]

            # cursor
            cv2.circle(output, (cx, cy), 10, current_color, -1)
            cv2.circle(output, (cx, cy), 18, (255, 255, 255), 2)

            # =========================
            # FIST CLEAR
            # =========================
            if closed_hand:
                if fist_frames < FIST_THRESHOLD:
                    fist_frames += 1

                cv2.putText(output, f"Clearing {fist_frames}/{FIST_THRESHOLD}",
                            (cx - 60, cy - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if fist_frames == FIST_THRESHOLD:
                    engine.clear()

            else:
                fist_frames = 0

            # =========================
            # UI INTERACTION
            # =========================
            if cy < 80:
                engine.reset_cursor()

                current_btn = None

                for btn in buttons:
                    x, y, bw, bh = btn["rect"]

                    if x < cx < x + bw and y < cy < y + bh:
                        current_btn = btn["name"]

                if current_btn == last_btn:
                    hover_frames += 1

                    if hover_frames > HOVER_THRESHOLD:

                        if current_btn == "CLEAR":
                            engine.clear()

                        elif current_btn in colors:
                            current_color = colors[current_btn]

                        elif current_btn == "QUIT":
                            running = False

                        hover_frames = 0
                        last_btn = None

                else:
                    hover_frames = 0
                    last_btn = current_btn

            else:
                if middle_up or closed_hand:
                    engine.reset_cursor()
                else:
                    engine.drawColor = current_color
                    engine.draw(cx, cy)

        else:
            engine.reset_cursor()
            fist_frames = 0

        # =========================
        # MERGE
        # =========================
        gray = cv2.cvtColor(engine.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)

        bg = cv2.bitwise_and(output, output, mask=mask)
        fg = cv2.bitwise_and(engine.canvas, engine.canvas, mask=cv2.bitwise_not(mask))

        final = cv2.add(bg, fg)

        # =========================
        # DRAW BUTTONS
        # =========================
        for btn in buttons:
            hover = (btn["name"] == last_btn)
            draw_button(final, btn, hover)

        cv2.imshow("Virtual Painter FINAL", final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

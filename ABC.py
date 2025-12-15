import cv2
import time
from ultralytics import YOLO
import pyttsx3

# Load YOLOv8 model (make sure yolov8n.pt is in your folder or specify path)
model = YOLO("yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

# Helper function for reliable speech on Windows
def speak(text):
    engine = pyttsx3.init(driverName="sapi5")
    engine.setProperty('volume', 1.0)
    engine.setProperty('rate', 175)
    voices = engine.getProperty('voices')
    if voices:
        engine.setProperty('voice', voices[0].id)  # choose first available voice
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# Cooldown control
last_spoken_times = {}
cooldown = 5.0  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            # Draw bounding boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Speak with cooldown
            now = time.time()
            last_time = last_spoken_times.get(label, 0)
            if conf > 0.6 and (now - last_time) >= cooldown:
                print(f"[SPEAK] I see a {label}")  # debug log
                speak(f"I see a {label}")
                last_spoken_times[label] = now

    cv2.imshow("Object Detection with Voice Feedback", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
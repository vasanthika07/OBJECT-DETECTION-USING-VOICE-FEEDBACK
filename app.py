import cv2
import pyttsx3
from collections import defaultdict
from ultralytics import YOLO

model = YOLO("yolov8x.pt")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

def speak(text):
    engine = pyttsx3.init(driverName="sapi5")
    engine.setProperty('volume', 1.0)
    engine.setProperty('rate', 175)
    voices = engine.getProperty('voices')
    if voices:
        engine.setProperty('voice', voices[0].id)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def position_bucket(box, frame_width):
    x1, _, x2, _ = box
    center = (x1 + x2) / 2.0
    if center < frame_width * 0.33:
        return "left"
    elif center > frame_width * 0.66:
        return "right"
    else:
        return "center"

def pluralize(label: str, count: int) -> str:
    irregulars = {
        "person": "people",
        "man": "men",
        "woman": "women",
        "child": "children",
        "mouse": "mice",
        "goose": "geese",
        "tooth": "teeth",
        "foot": "feet",
        "bus": "buses",
    }
    if count == 1:
        return label
    if label in irregulars:
        return irregulars[label]
    if label.endswith("y") and label[-2] not in "aeiou":
        return label[:-1] + "ies"
    if label.endswith(("s", "x", "z", "ch", "sh")):
        return label + "es"
    return label + "s"

last_announced = set()  # what we have already spoken

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    grouped = defaultdict(int)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if conf > 0.6:
                pos = position_bucket((x1, y1, x2, y2), frame.shape[1])
                grouped[(label, pos)] += 1

    current_announced = set(grouped.keys())

    # Speak only for new entries (not in last_announced)
    new_objects = current_announced - last_announced
    if new_objects:
        messages = []
        for (label, pos) in new_objects:
            count = grouped[(label, pos)]
            if count == 1:
                messages.append(f"{label} on the {pos}")
            else:
                messages.append(f"{count} {pluralize(label, count)} on the {pos}")
        sentence = ", ".join(messages)
        print("[SPEAK]", sentence)
        speak(sentence)

    # Update last_announced only with objects present in this frame
    last_announced = current_announced

    cv2.imshow("Object Detection with Voice Feedback", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

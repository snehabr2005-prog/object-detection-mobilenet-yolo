import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load models
print("Loading models... Please wait")
yolo_model = YOLO('yolov8n.pt') # Will auto-download first time
mobilenet_model = MobileNetV2(weights='imagenet')
print("Models Loaded!")

# Open Camera
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
current_mode = "mobilenet" # default mode

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    # MODE 1: MobileNetV2 Classification
    if current_mode == "mobilenet":
        # Preprocess for MobileNet
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        preds = mobilenet_model.predict(img, verbose=0)
        label = decode_predictions(preds, top=1)[0][0]

        # label = (class_id, name, confidence)
        text = f"{label[1]}: {label[2]*100:.1f}%"
        cv2.putText(display_frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Mode: MobileNet (Press 'y' for YOLO)", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # MODE 2: YOLOv8 Detection
    else:
        results = yolo_model(frame, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = yolo_model.names[cls]

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(display_frame, "Mode: YOLOv8 (Press 'm' for MobileNet)", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("CNN Image Recognition - Press m/y/q", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        current_mode = "mobilenet"
        print("Switched to MobileNet")
    elif key == ord('y'):
        current_mode = "yolo"
        print("Switched to YOLO")

cap.release()
cv2.destroyAllWindows()
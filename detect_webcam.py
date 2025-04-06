import cv2
from ultralytics import YOLO


model = YOLO("yolov5n.pt")

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        
        if class_name == 'car':
            print(0)
        elif class_name == 'motorcycle':
            print(1)
        elif class_name == 'bicycle':
            print(2)

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

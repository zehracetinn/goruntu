import cv2
from ultralytics import YOLO

# model yükle
model = YOLO("yolov8n.pt")

# webcam aç
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection
    results = model(frame)

    # bounding box çiz
    annotated_frame = results[0].plot()

    # ekrana göster
    cv2.imshow("YOLO Detection", annotated_frame)

    # q tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
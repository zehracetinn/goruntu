import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.track(
    source=0,
    show=True,
    tracker="bytetrack.yaml"
)
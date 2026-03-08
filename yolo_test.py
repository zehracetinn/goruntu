from ultralytics import YOLO

# modeli yükle
model = YOLO("yolov8n.pt")

# görüntü üzerinde çalıştır
results = model("test3.jpg")

# sonucu göster
results[0].show()
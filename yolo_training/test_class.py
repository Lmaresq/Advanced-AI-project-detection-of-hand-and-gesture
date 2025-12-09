
from ultralytics import YOLO
m = YOLO("backend/models/yolo11n.pt")
print(m.model.names)

## Code to train model

import cv2
from ultralytics import YOLO

# Load YOLOv8 model 
model = YOLO('yolov8n.pt')

# Train the model
model.train(data='/Users/clairekaufman/Downloads/MAtomics-Data/data_config.yaml', epochs=50, imgsz=640)

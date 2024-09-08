import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ultralytics import YOLO


model = YOLO("yolov8n.pt")

# will throw an exception if false
model._check_is_pytorch_model()

data_yaml_path = "Skin Cancer Recogniser.v2i.yolov8/data.yaml"

# Use 'cpu' for device since you don't have CUDA available
model.train(data=data_yaml_path,
            epochs=100,
            imgsz=100,
            device='cpu')
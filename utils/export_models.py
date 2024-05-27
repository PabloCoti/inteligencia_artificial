from ultralytics import YOLO

# Initialize models
models = [YOLO("../models/yolov8l.pt"), YOLO("../models/bancopoly.pt")]

# Export to onnx
for model in models:
    model.export(format="onnx")

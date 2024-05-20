from ultralytics import YOLO
import os

# Get all .pt files in the models directory
model_files = [file for file in os.listdir("models") if file.endswith(".pt")]

# Initialize models
models = [YOLO(os.path.join("models", file)) for file in model_files]

# Export to onnx
for model in models:
    model.export(format="onnx")

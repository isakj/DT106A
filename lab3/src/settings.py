import os

# Working directory and dataset paths (lab2/src -> lab2/dataset, lab2/model)
CWDIR = os.path.dirname(__file__)
DATASET_DIR = os.path.normpath(os.path.join(CWDIR, "..", "dataset"))
DATASET_YAML = os.path.join(DATASET_DIR, "dataset.yaml")
MODEL_DIR = os.path.normpath(os.path.join(CWDIR, "..", "model"))

# Hyperparameters and settings for network training
# To do: Adjust the split ratios as needed for your dataset, ensuring they sum to 1.0
SPLIT_RATIOS = {
    "train": 0.7,
    "valid": 0.1,
    "test": 0.2,
}

# Training (3_train.py: default epochs, min 50 enforced in script)
EPOCHS = 100
DEVICE = "cpu"

# Ultralytics pretrained weights name (hub / package); 3_train loads yolov8n.pt explicitly
YOLO_MODEL = "yolov8n.pt"
# Run subfolder under MODEL_DIR (weights/best.pt for inference)
MODEL_NAME = "baseline_nano-3"

# To do: Adjust the save period to save only the best model based on validation performance
MODEL_SAVE_PERIOD = 1

# Model prediction threshold for output bounding boxes
PRED_THRESHOLD = 0.5
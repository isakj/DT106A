import os

# Working directory and dataset paths
CWDIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(CWDIR, "../../dataset")
DATASET_YAML = os.path.join(DATASET_DIR, "dataset.yaml")
MODEL_DIR = os.path.join(CWDIR, "../../model")

# Hyperparameters and settings for network training
# To do: Adjust the split ratios as needed for your dataset, ensuring they sum to 1.0
SPLIT_RATIOS = {
    "train": 0.0,
    "valid": 0.0,
    "test": 0.0,
}

# To do: Adjust the number of epochs and device for training based on your setup and requirements
EPOCHS = 5
DEVICE = "cpu"

# Model settings
# To do: Set the YOLO_MODEL to the appropriate model architecture you want to use for training, i.e., YOLOv8 Nano.
YOLO_MODEL = ""  # specify the YOLO model architecture to use for training
MODEL_NAME = ''  # name for the trained model, used in saving outputs

# To do: Adjust the save period to save only the best model based on validation performance
MODEL_SAVE_PERIOD = 1

# Model prediction threshold for output bounding boxes
PRED_THRESHOLD = 0.5
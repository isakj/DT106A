from ultralytics import YOLO
import os
from settings import DEVICE, MODEL_NAME, MODEL_SAVE_PERIOD, EPOCHS, MODEL_DIR, DATASET_YAML, YOLO_MODEL

print("Starting training script...")

model = YOLO(os.path.join(MODEL_DIR,YOLO_MODEL))
print("Model loaded, starting training...")

"""Call the train method on the model using model.train() with the appropriate parameters to start training the model on the training dataset.

TODO:
- Parse the dataset YAML file to get the path to the training images and labels.
- Set the appropriate parameters for training, such as epochs, device.
- Set the model name and project directory to save the trained model and training outputs, save only the best model based on validation performance.

"""

print("Training finished")
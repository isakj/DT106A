import os
import random
import shutil
from settings import DATASET_DIR, SPLIT_RATIOS

IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

def check_directories():
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(LABELS_DIR):
        print("Images or labels directory not found. Please ensure the dataset is organized correctly."
        )
        return False
    return True

def delete_existing_splits():
    for split in SPLIT_RATIOS:
        split_dir = f"{DATASET_DIR}/{split}"
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
            print(f"Deleted existing {split} split.")

def create_dirs():
    for split in SPLIT_RATIOS:
        os.makedirs(f"{DATASET_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{DATASET_DIR}/{split}/labels", exist_ok=True)

def split_data():
    images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
    random.shuffle(images)

    n = len(images)
    train_end = int(n * SPLIT_RATIOS["train"])
    valid_end = train_end + int(n * SPLIT_RATIOS["valid"])

    splits = {
        "train": images[:train_end],
        "valid": images[train_end:valid_end],
        "test": images[valid_end:],
    }

    for split, files in splits.items():
        for img in files:
            label = img.replace(".jpg", ".txt")

            shutil.copy(
                os.path.join(IMAGES_DIR, img),
                f"{DATASET_DIR}/{split}/images/{img}",
            )
            shutil.copy(
                os.path.join(LABELS_DIR, label),
                f"{DATASET_DIR}/{split}/labels/{label}",
            )

    print("Dataset split complete!")

def count_files():
    for split in SPLIT_RATIOS:
        img_count = len(os.listdir(f"{DATASET_DIR}/{split}/images"))
        label_count = len(os.listdir(f"{DATASET_DIR}/{split}/labels"))
        print(f"{split.capitalize()} - Images: {img_count}, Labels: {label_count}")

if __name__ == "__main__":
    
    """ Complete the main function by using the above implemented functions to perform the following steps:
    TODO:
    1. Check if the images and labels directories exist. If not, print an error message and exit.
    2. If the directories exist, delete any existing train/valid/test splits to avoid confusion.
    3. Create new directories for the train, valid, and test splits.
    4. Split the dataset according to the specified ratios and copy the images and labels to the corresponding directories.
    5. Count and print the number of images and labels in each split to verify the distribution.        
    """
    exit()

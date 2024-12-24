import os
import shutil
import glob
import numpy as np

# Paths (adjust as needed)
images_train_dir = "potholes/output/train/images"
labels_train_dir = "potholes/output/train/labels"

images_val_dir = "potholes/output/val/images"
labels_val_dir = "potholes/output/val/labels"

# Make sure the val directories exist
os.makedirs(images_val_dir, exist_ok=True)
os.makedirs(labels_val_dir, exist_ok=True)

# Ratio of images to move to val
val_ratio = 0.2  # 20% to val

# Gather all images in train
image_paths = glob.glob(os.path.join(images_train_dir, "*.png")) + glob.glob(os.path.join(images_train_dir, "*.jpg"))

# Shuffle them randomly
np.random.shuffle(image_paths)

# Determine how many images to move
val_count = int(len(image_paths) * val_ratio)
val_paths = image_paths[:val_count]

for img_path in val_paths:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(labels_train_dir, base_name + ".txt")

    # Move image to val folder
    shutil.move(img_path, os.path.join(images_val_dir, os.path.basename(img_path)))

    # Move label to val folder if it exists
    if os.path.exists(label_path):
        shutil.move(label_path, os.path.join(labels_val_dir, os.path.basename(label_path)))

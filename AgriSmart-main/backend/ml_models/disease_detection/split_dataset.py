import os
import shutil
import random

# Paths
SOURCE_DIR = "PlantVillage"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

# Create train and test folders if not exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Split ratio
SPLIT_RATIO = 0.8

# Loop through each disease folder
for disease in os.listdir(SOURCE_DIR):
    disease_path = os.path.join(SOURCE_DIR, disease)

    if not os.path.isdir(disease_path):
        continue

    images = os.listdir(disease_path)
    random.shuffle(images)

    split_point = int(len(images) * SPLIT_RATIO)

    train_images = images[:split_point]
    test_images = images[split_point:]

    # Create class folders
    os.makedirs(os.path.join(TRAIN_DIR, disease), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, disease), exist_ok=True)

    # Copy images
    for img in train_images:
        shutil.copy(
            os.path.join(disease_path, img),
            os.path.join(TRAIN_DIR, disease, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(disease_path, img),
            os.path.join(TEST_DIR, disease, img)
        )

print("✅ Dataset split completed successfully!")
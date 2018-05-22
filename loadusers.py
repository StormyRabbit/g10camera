import os
TRAINING_DATA_PATH = "users/"

for folder in os.listdir(TRAINING_DATA_PATH):
    if os.path.isdir(TRAINING_DATA_PATH + folder):
        for image in os.listdir(TRAINING_DATA_PATH + folder):
            print("USER:", image.split("_")[0])

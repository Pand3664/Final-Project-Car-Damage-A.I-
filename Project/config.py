import os
import numpy as np

BASE_DIR = '/content/Project'
DATA_CSV = os.path.join(BASE_DIR, "data.csv")
IMAGE_DIR = "/content/Image"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "Models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "Plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
IMG_CHANNELS = 3

TEST_SIZE = 0.2
VAL_SIZE = 0.15
RANDOM_STATE = 42

DL_EPOCHS = 20
DL_BATCH_SIZE = 32
DL_LR = 1e-4
DL_DROPOUT = 0.40
FINE_TUNE_AT = 100

CLASS_LABELS = [
    "unknown",
    "bumper_scratch",
    "door_scratch",
    "bumper_dent",
    "door_dent",
    "head_lamp",
    "tail_lamp",
    "glass_shatter",
]

NUM_CLASSES = len(CLASS_LABELS)

BASE_SEVERITY = {
    "unknown": 1,
    "bumper_scratch": 2,
    "door_scratch": 2,
    "bumper_dent": 3,
    "door_dent": 3,
    "head lamp": 4,
    "head_lamp": 4,
    "tail lamp": 4,
    "tail_lamp": 4,
    "glass shatter": 5,
    "glass_shatter": 5,
}

SEVERITY_DESCRIPTIONS = {
    1: "Unknown - No significant damages were detected",
    2: "Low - cosmetic damage, no structural",
    3: "Medium - structural damage present; repairs advised",
    4: "High - safety concerns, component damgages, urget repairs needed",
    5: "Critical - immediate repairs needed, vehicle cannot be driven for safety concerns",
}

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

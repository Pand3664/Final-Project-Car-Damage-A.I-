import config
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_image_pil(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def preprocess_image(img: Image.Image, size=config.IMG_SIZE) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((size[1], size[0]))
    return np.array(img, dtype=np.float32) / 255.0

def extract_features(img_array: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return hog(
        gray,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PIXELS_PER_CELL,
        cells_per_block=config.HOG_CELLS_PER_BLOCK,
        feature_vector=True,
    )

def extract_color_histogram(img_array: np.ndarray, bins: int = 32) -> np.ndarray:
    hists = []
    for ch in range(3):
        h, _ = np.histogram(img_array[:, :, ch] * 255, bins=bins, range=(0, 256))
        hists.append(h.astype(np.float32) / (h.sum() + 1e-8))
    return np.concatenate(hists)

def extract_combined_features(img_array: np.ndarray) -> np.ndarray:
    return np.concatenate([
        extract_features(img_array),
        extract_color_histogram(img_array),
    ])

class VehicleDamageDataset:
    def __init__(self, csv_file: str = config.DATA_CSV, image_dir: str = config.IMAGE_DIR):
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.df = None
        self.images = []
        self.labels = []
        self.le = LabelEncoder()
        self.loaded = False

    def load(self, skip_unknown: bool = False):
        self.df = pd.read_csv(self.csv_file)
        cols = {c.lower().strip(): c for c in self.df.columns}
        img_col = cols.get("image_id") or cols.get("image") or cols.get("filename")
        lbl_col = cols.get("classes") or cols.get("class") or cols.get("label")

        if skip_unknown:
            self.df = self.df[self.df[lbl_col].astype(str).str.lower().str.strip() != "unknown"].reset_index(drop=True)

        images, labels = [], []
        print(f"DEBUG: Starting load from {self.image_dir}")
        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Loading"):
            img_path = os.path.join(self.image_dir, os.path.basename(str(row[img_col])))
            if i < 3: # Debug: print first 3 paths
                print(f"DEBUG: Inspecting path -> {img_path}")
                print(f"DEBUG: File exists? -> {os.path.exists(img_path)}")

            img = load_image_pil(img_path)
            if img is not None:
                images.append(preprocess_image(img))
                labels.append(str(row[lbl_col]).strip())

        if len(images) == 0:
            raise ValueError(f"No images loaded from {self.image_dir}. Check paths and CSV content.")

        self.images, self.labels = np.array(images, dtype=np.float32), np.array(labels)
        self.le.fit(self.labels)
        self.loaded = True
        return self

    def get_classical_splits(self):
        print("Extracting features...")
        X = np.array([extract_combined_features(img) for img in tqdm(self.images)])
        y = self.le.transform(self.labels)

        if len(X) < 2:
            raise ValueError("Not enough samples to split into train and test sets.")

        splits = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y)
        return list(splits) + [self.le]

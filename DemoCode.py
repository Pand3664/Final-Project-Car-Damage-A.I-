# =========================================================
# CAR DAMAGE SEVERITY AI — FULL FIXED LOCAL VERSION
# =========================================================

import os
import sys
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from skimage.feature import hog

# =========================================================
# PROJECT PATH (FIXED FOR MAC / LOCAL)
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

DATA_CSV = os.path.join(SCRIPT_DIR, "data.csv")
IMAGE_DIR = os.path.join(SCRIPT_DIR, "Image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "Models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "Plots")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================================================
# LABELS + SEVERITY SYSTEM
# =========================================================
CLASS_LABELS = [
    "unknown",
    "Bumper_Scratch",
    "Door_Scratch",
    "Bumper_Dent",
    "Door_Dent",
    "Head lamp",
    "Tail lamp",
    "Glass Shatter",
]

BASE_SEVERITY = {
    "unknown": 1,
    "Bumper_Scratch": 2,
    "Door_Scratch": 2,
    "Bumper_Dent": 3,
    "Door_Dent": 3,
    "Head lamp": 4,
    "Tail lamp": 4,
    "Glass Shatter": 5,
}

SEVERITY_DESCRIPTIONS = {
    1: "Unknown - No significant damage",
    2: "Low - cosmetic damage",
    3: "Medium - structural damage",
    4: "High - safety concern",
    5: "Critical - do not drive",
}

# =========================================================
# IMAGE UTILITIES
# =========================================================
def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        return None

def preprocess(img):
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))
    return np.array(img, dtype=np.float32) / 255.0

def hog_features(img):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True
    )

def color_hist(img):
    feats = []
    for i in range(3):
        h, _ = np.histogram(img[:, :, i] * 255, bins=32, range=(0, 256))
        feats.append(h / (h.sum() + 1e-8))
    return np.concatenate(feats)

def extract_features(img):
    return np.concatenate([hog_features(img), color_hist(img)])

# =========================================================
# DATASET LOADER
# =========================================================
class Dataset:
    def __init__(self):
        self.df = None
        self.images = []
        self.labels = []
        self.le = LabelEncoder()

    def load(self):
        self.df = pd.read_csv(DATA_CSV)

        cols = {c.lower(): c for c in self.df.columns}
        img_col = cols.get("image_id") or cols.get("image") or cols.get("filename")
        lbl_col = cols.get("class") or cols.get("classes") or cols.get("label")

        imgs, lbls = [], []

        for _, row in self.df.iterrows():
            path = os.path.join(IMAGE_DIR, os.path.basename(str(row[img_col])))
            img = load_image(path)

            if img:
                imgs.append(preprocess(img))
                lbls.append(str(row[lbl_col]).strip())

        self.images = np.array(imgs)
        self.labels = np.array(lbls)
        self.le.fit(self.labels)
        return self

    def split(self):
        X = np.array([extract_features(i) for i in self.images])
        y = self.le.transform(self.labels)

        return train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        ), self.le

# =========================================================
# MODEL
# =========================================================
def build_model():
    return VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=50)),
            ("svm", make_pipeline(StandardScaler(), SVC(probability=True))),
            ("gb", HistGradientBoostingClassifier(max_iter=20))
        ],
        voting="soft"
    )

# =========================================================
# SEVERITY SCORING
# =========================================================
def severity(pred_class, conf):
    base = BASE_SEVERITY.get(pred_class, 1)
    score = int(np.clip(round(base + (conf - 0.5)), 1, 5))
    return {
        "class": pred_class,
        "score": score,
        "label": SEVERITY_DESCRIPTIONS[score],
        "confidence": conf
    }

def print_report(res):
    bar = "#" * res["score"] + "-" * (5 - res["score"])
    print(f"\n{res['class']} | {res['score']}/5 [{bar}]")
    print(res["label"])

# =========================================================
# TRAINING
# =========================================================
def train():
    ds = Dataset().load()
    (Xtr, Xte, ytr, yte), le = ds.split()

    model = build_model()
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)

    print(f"\nAccuracy: {acc*100:.2f}%")

    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "encoder.pkl"))

# =========================================================
# LOAD MODEL
# =========================================================
def load_model():
    m = os.path.join(MODEL_DIR, "model.pkl")
    e = os.path.join(MODEL_DIR, "encoder.pkl")

    if not os.path.exists(m):
        return None, None

    return joblib.load(m), joblib.load(e)

# =========================================================
# PREDICT
# =========================================================
def predict_image(path, model, le):
    img = load_image(path)
    if img is None:
        return {"error": "bad image"}

    x = preprocess(img)
    feat = extract_features(x).reshape(1, -1)

    probs = model.predict_proba(feat)[0]
    idx = np.argmax(probs)

    cls = le.inverse_transform([idx])[0]
    conf = float(probs[idx])

    return severity(cls, conf)

# =========================================================
# DEMO
# =========================================================
def demo(n=5):
    model, le = load_model()
    if model is None:
        print("Run training first.")
        return

    df = pd.read_csv(DATA_CSV)

    cols = {c.lower(): c for c in df.columns}
    img_col = cols.get("image_id") or cols.get("image") or cols.get("filename")
    lbl_col = cols.get("class") or cols.get("classes") or cols.get("label")

    sample = df.sample(n=min(n, len(df)))

    for _, r in sample.iterrows():
        path = os.path.join(IMAGE_DIR, os.path.basename(str(r[img_col])))
        res = predict_image(path, model, le)

        if "error" in res:
            continue

        print_report(res)

        plt.imshow(Image.open(path))
        plt.title(res["class"])
        plt.axis("off")
        plt.show()

# =========================================================
# RUN PROJECT
# =========================================================
if __name__ == "__main__":
    print("\n1. TRAIN MODEL")
    print("2. RUN DEMO")

    choice = input("Select option: ")

    if choice == "1":
        train()
    elif choice == "2":
        demo()
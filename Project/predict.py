import config
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from data_loader import load_image_pil, preprocess_image, extract_combined_features
from severity_scorer import compute_severity

def load_model():
    m_path = os.path.join(config.MODEL_DIR, "best_classical_model.pkl")
    l_path = os.path.join(config.MODEL_DIR, "label_encoder.pkl")
    if not os.path.exists(m_path): return None, None
    return joblib.load(m_path), joblib.load(l_path)

def predict_image(path, model, le):
    img = load_image_pil(path)
    if img is None: return {"error": "bad image"}
    x = preprocess_image(img)
    p = model.predict_proba(extract_combined_features(x).reshape(1, -1))[0]
    idx = int(np.argmax(p))
    return compute_severity(le.inverse_transform([idx])[0], float(p[idx]))

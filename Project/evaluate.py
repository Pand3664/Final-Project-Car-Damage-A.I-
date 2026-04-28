import config
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def evaluate_classifier(y_true, y_pred, le, name="Model"):
    acc = accuracy_score(y_true, y_pred)
    print(f"[{name}] Accuracy: {acc*100:.2f}%")
    return {"Accuracy": acc}

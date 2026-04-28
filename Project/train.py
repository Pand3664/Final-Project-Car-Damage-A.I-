import os
import joblib
import config
from data_loader import VehicleDamageDataset
import evaluate as ev
from models import CLASSICAL_MODELS

def train_classical(skip_unknown: bool):
    ds = VehicleDamageDataset().load(skip_unknown=skip_unknown)
    X_tr, X_te, y_tr, y_te, le = ds.get_classical_splits()
    joblib.dump(le, os.path.join(config.MODEL_DIR, 'label_encoder.pkl'))

    print(f'\nTraining Classical Models (skip_unknown={skip_unknown})...')
    for name, model in CLASSICAL_MODELS.items():
        print(f'  Fitting {name}...')
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        ev.evaluate_classifier(y_te, y_pred, le, name)
        joblib.dump(model, os.path.join(config.MODEL_DIR, 'best_classical_model.pkl'))
    print('Training complete.')

if __name__ == '__main__':
    train_classical(skip_unknown=True)

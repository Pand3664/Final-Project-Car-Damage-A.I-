import config
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def build_ensemble():
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=50, random_state=config.RANDOM_STATE)),
        ("svm", make_pipeline(StandardScaler(), SVC(probability=True, random_state=config.RANDOM_STATE))),
        ("gb", HistGradientBoostingClassifier(max_iter=20, random_state=config.RANDOM_STATE))
    ]
    return VotingClassifier(estimators, voting="soft")

CLASSICAL_MODELS = {"Soft-Voting Ensemble": build_ensemble()}

# extract_params.py (run in kaggle_clone)
import joblib
import os

BASE_DIR = "/run/media/biprarshi/COMMON/files/AI/MINST_visualizer/digit_classifier/backend"

models = {
    'DT': 'models/decision_tree/sklearn_DT.pkl',
    'RF': 'models/random_forest/random_forest_sklearn.pkl',
    'GB': 'models/random_forest/gbdt_sklearn.pkl',
}

params = {}
for name, path in models.items():
    full_path = os.path.join(BASE_DIR, path)
    model = joblib.load(full_path)
    
    if name == 'DT':
        params['DT'] = {
            'max_depth': model.max_depth,
            'min_samples_split': model.min_samples_split,
            'min_samples_leaf': model.min_samples_leaf,
            'random_state': model.random_state,
        }
    elif name == 'RF':
        params['RF'] = {
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'min_samples_split': model.min_samples_split,
            'min_samples_leaf': model.min_samples_leaf,
            'random_state': model.random_state,
        }
    elif name == 'GB':
        params['GB'] = {
            'n_estimators': model.n_estimators,
            'learning_rate': model.learning_rate,
            'max_depth': model.max_depth,
            'min_samples_split': model.min_samples_split,
            'min_samples_leaf': model.min_samples_leaf,
            'random_state': model.random_state,
        }
    
    print(f"{name}: {params[name]}")

# Save for Kaggle
import json
with open('/tmp/model_params.json', 'w') as f:
    json.dump(params, f, default=str)
print(f"\nSaved to /tmp/model_params.json")
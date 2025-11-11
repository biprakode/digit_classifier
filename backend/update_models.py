# extract_in_old_env.py
import joblib
import numpy as np
import json
import os

BASE_DIR = "/run/media/biprarshi/COMMON/files/AI/MINST_visualizer/digit_classifier/backend"
OUT_DIR = "/tmp/model_internals"
os.makedirs(OUT_DIR, exist_ok=True)

files = [
    'models/log_reg/logreg_sklearn.pkl',
    'models/naive_bayes/naive_bayes_sklearn.pkl',
    'models/decision_tree/sklearn_DT.pkl',
    'models/random_forest/random_forest_sklearn.pkl',
    'models/random_forest/gbdt_sklearn.pkl',
    'models/KNN/knn_sklearn.pkl',
]

def convert(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

for rel_path in files:
    path = os.path.join(BASE_DIR, rel_path)
    name = os.path.basename(rel_path).replace(".pkl", "")
    
    print(f"Extracting: {name}")
    try:
        model = joblib.load(path)
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        continue

    data = {}

    if "logreg" in name:
        data["coef_"] = model.coef_
        data["intercept_"] = model.intercept_
        data["classes_"] = model.classes_
        data["type"] = "logreg"

    elif "naive" in name:
        data["feature_log_prob_"] = model.feature_log_prob_
        data["class_log_prior_"] = model.class_log_prior_
        data["classes_"] = model.classes_
        data["type"] = "nb"

    elif "DT" in name:
        tree = model.tree_
        data["type"] = "dt"
        data["feature"] = tree.feature
        data["threshold"] = tree.threshold
        data["children_left"] = tree.children_left
        data["children_right"] = tree.children_right
        data["value"] = tree.value
        data["n_classes"] = model.n_classes_
        data["n_features_in_"] = model.n_features_in_
        data["classes_"] = model.classes_

    elif "random_forest" in name:
        data["type"] = "rf"
        data["n_estimators"] = len(model.estimators_)
        data["estimators"] = []
        for est in model.estimators_:
            t = est.tree_
            data["estimators"].append({
                "feature": t.feature,
                "threshold": t.threshold,
                "children_left": t.children_left,
                "children_right": t.children_right,
                "value": t.value,
            })
        data["classes_"] = model.classes_
        data["n_features_in_"] = model.n_features_in_

    elif "gbdt" in name:
        data["type"] = "gb"
        data["estimators"] = []
        data["learning_rate"] = model.learning_rate
        data["n_estimators"] = model.n_estimators
        data["init_"] = model.init_.__class__.__name__  # e.g., "LogisticRegression"
        
        for stage in model.estimators_:
            stage_trees = []
            for tree in stage:
                t = tree.tree_  # tree is DecisionTreeRegressor
                stage_trees.append({
                    "feature": t.feature,
                    "threshold": t.threshold,
                    "children_left": t.children_left,
                    "children_right": t.children_right,
                    "value": t.value,
                })
            data["estimators"].append(stage_trees)
        
        data["classes_"] = model.classes_
        data["n_features_in_"] = model.n_features_in_

    elif "knn" in name:
        data["type"] = "knn"
        data["_fit_X"] = model._fit_X
        data["_y"] = model._y
        data["n_neighbors"] = model.n_neighbors
        data["classes_"] = model.classes_

    # Save with conversion
    out_path = os.path.join(OUT_DIR, f"{name}.json")
    try:
        with open(out_path, 'w') as f:
            json.dump(data, f, default=convert)
        print(f"Saved internals: {out_path}")
    except Exception as e:
        print(f"Failed to save {name}: {e}")
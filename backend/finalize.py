# rebuild_in_modern.py
import json
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import Tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

IN_DIR = "/tmp/model_internals"
BASE_DIR = "/run/media/biprarshi/COMMON/files/AI/MINST_visualizer/digit_classifier/backend"

mapping = {
    "logreg_sklearn.json": "models/log_reg/logreg_sklearn.pkl",
    "naive_bayes_sklearn.json": "models/naive_bayes/naive_bayes_sklearn.pkl",
    "sklearn_DT.json": "models/decision_tree/sklearn_DT.pkl",
    "random_forest_sklearn.json": "models/random_forest/random_forest_sklearn.pkl",
    "gbdt_sklearn.json": "models/random_forest/gbdt_sklearn.pkl",
    "knn_sklearn.json": "models/KNN/knn_sklearn.pkl",
}

for json_file, pkl_path in mapping.items():
    json_path = os.path.join(IN_DIR, json_file)
    final_path = os.path.join(BASE_DIR, pkl_path)
    if not os.path.exists(json_path):
        print(f"Missing: {json_file}")
        continue

    print(f"Rebuilding: {json_file}")
    with open(json_path) as f:
        data = json.load(f)

    model = None
    if data["type"] == "logreg":
        model = LogisticRegression()
        model.coef_ = np.array(data["coef_"])
        model.intercept_ = np.array(data["intercept_"])
        model.classes_ = np.array(data["classes_"])

    elif data["type"] == "nb":
        model = MultinomialNB()
        model.feature_log_prob_ = np.array(data["feature_log_prob_"])
        model.class_log_prior_ = np.array(data["class_log_prior_"])
        model.classes_ = np.array(data["classes_"])

    elif data["type"] == "dt":
        n_nodes = len(data["feature"])
        n_classes = len(data["classes_"])
        tree = Tree(n_features=data["n_features_in_"], n_classes=np.array([n_classes]), n_outputs=1)
        tree.capacity = n_nodes
        tree.node_count = n_nodes
        # Use __setstate__ to set internal arrays (bypasses writability)
        tree.__setstate__({
            'node_count': n_nodes,
            'capacity': n_nodes,
            'feature': np.array(data["feature"], dtype=np.int64),
            'threshold': np.array(data["threshold"], dtype=np.float64),
            'children_left': np.array(data["children_left"], dtype=np.int64),
            'children_right': np.array(data["children_right"], dtype=np.int64),
            'value': np.array(data["value"], dtype=np.float64).reshape(n_nodes, 1, n_classes),
        })
        model = DecisionTreeClassifier()
        model.tree_ = tree
        model.classes_ = np.array(data["classes_"])
        model.n_classes_ = n_classes
        model.n_features_in_ = data["n_features_in_"]

    elif data["type"] == "rf":
        model = RandomForestClassifier(n_estimators=data["n_estimators"])
        model.classes_ = np.array(data["classes_"])
        model.n_classes_ = len(data["classes_"])
        model.n_features_in_ = data["n_features_in_"]
        n_classes = model.n_classes_
        model.estimators_ = []
        for est_data in data["estimators"]:
            n_nodes = len(est_data["feature"])
            tree = Tree(n_features=data["n_features_in_"], n_classes=np.array([n_classes]), n_outputs=1)
            tree.__setstate__({
                'node_count': n_nodes,
                'capacity': n_nodes,
                'feature': np.array(est_data["feature"], dtype=np.int64),
                'threshold': np.array(est_data["threshold"], dtype=np.float64),
                'children_left': np.array(est_data["children_left"], dtype=np.int64),
                'children_right': np.array(est_data["children_right"], dtype=np.int64),
                'value': np.array(est_data["value"], dtype=np.float64).reshape(n_nodes, 1, n_classes),
            })
            est = DecisionTreeClassifier()
            est.tree_ = tree
            est.classes_ = model.classes_
            est.n_classes_ = n_classes
            est.n_features_in_ = data["n_features_in_"]
            model.estimators_.append(est)

    elif data["type"] == "gb":
        model = GradientBoostingClassifier(n_estimators=data["n_estimators"], learning_rate=data["learning_rate"])
        model.classes_ = np.array(data["classes_"])
        model.n_classes_ = len(data["classes_"])
        model.n_features_in_ = data["n_features_in_"]
        estimators = []
        for stage in data["estimators"]:
            stage_trees = []
            for tree_data in stage:
                n_nodes = len(tree_data["feature"])
                tree = Tree(n_features=data["n_features_in_"], n_classes=np.array([1]), n_outputs=1)
                tree.__setstate__({
                    'node_count': n_nodes,
                    'capacity': n_nodes,
                    'feature': np.array(tree_data["feature"], dtype=np.int64),
                    'threshold': np.array(tree_data["threshold"], dtype=np.float64),
                    'children_left': np.array(tree_data["children_left"], dtype=np.int64),
                    'children_right': np.array(tree_data["children_right"], dtype=np.int64),
                    'value': np.array(tree_data["value"], dtype=np.float64).reshape(n_nodes, 1, 1),
                })
                regressor = DecisionTreeRegressor()
                regressor.tree_ = tree
                regressor.n_features_in_ = data["n_features_in_"]
                stage_trees.append(regressor)
            estimators.append(np.array(stage_trees))
        model.estimators_ = np.array(estimators, dtype=object)
        model.init_ = None

    elif data["type"] == "knn":
        model = KNeighborsClassifier(n_neighbors=data["n_neighbors"])
        model._fit_X = np.array(data["_fit_X"])
        model._y = np.array(data["_y"])
        model.classes_ = np.array(data["classes_"])

    if model:
        joblib.dump(model, final_path)
        print(f"SAVED: {pkl_path}")
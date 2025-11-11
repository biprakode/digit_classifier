import os

# Check if model files exist
model_files = {
    "sklearn_logreg": "models/log_reg/logreg_sklearn.pkl",
    "custom_logreg": "models/log_reg/logreg_custom_regular.npz",
    "sklearn_nb": "models/naive_bayes/naive_bayes_sklearn.pkl",
    "custom_nb": "models/naive_bayes/naive_bayes_custom.npz",
    "sklearn_DT": "models/decision_tree/sklearn_DT.pkl",
    "custom_DT": "models/decision_tree/custom_DT.pkl",
    "sklearn_KNN": "models/KNN/knn_sklearn.pkl",
    "sklearn_RF": "models/random_forest/random_forest_sklearn.pkl",
    "custom_RF": "models/random_forest/forest.pkl",
    "sklearn_GB": "models/random_forest/gbdt_sklearn.pkl",
    "custom_GB": "models/random_forest/gbdt_model.pkl",
    "custom_DNN_1": "models/dnn_cnn/my_dnn_1.pkl",
    "custom_DNN_2": "models/dnn_cnn/my_dnn_2.pkl",
    "custom_DNN_3": "models/dnn_cnn/my_dnn_3.pkl",
    "torch_DNN_1": "models/dnn_cnn/torch_dnn_1.pt",
    "torch_DNN_2": "models/dnn_cnn/torch_dnn_2.pt",
    "torch_DNN_3": "models/dnn_cnn/torch_dnn_3.pt",
    "torch_CNN_1": "models/dnn_cnn/torch_cnn_1.pt",
    "torch_CNN_2": "models/dnn_cnn/torch_cnn_2.pt",
}

print("Checking model file paths...")
print("="*60)

for model_name, path in model_files.items():
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {model_name:20s} -> {path}")

print("="*60)
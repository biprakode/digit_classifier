"""Verify all expected model weight files exist on disk."""
import os
import pytest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILES = {
    "sklearn_logreg": "models/log_reg/retrained_sklearn_logreg.pkl",
    "custom_logreg": "models/log_reg/logreg_custom_regular.npz",
    "sklearn_nb": "models/naive_bayes/retrained_sklearn_nb.pkl",
    "custom_nb": "models/naive_bayes/naive_bayes_custom.npz",
    "sklearn_DT": "models/decision_tree/retrained_sklearn_DT.pkl",
    "custom_DT": "models/decision_tree/custom_DT.pkl",
    "sklearn_SVM": "models/SVM/retrained_sklearn_SVM.pkl",
    "sklearn_RF": "models/random_forest/retrained_sklearn_RF.pkl",
    "custom_RF": "models/random_forest/forest.pkl",
    "sklearn_GB": "models/random_forest/retrained_sklearn_GB.pkl",
    "custom_GB": "models/random_forest/gbdt_model.pkl",
    "sklearn_KNN": "models/KNN/retrained_sklearn_KNN.pkl",
    "custom_DNN_1": "models/dnn_cnn/my_dnn_1.pkl",
    "custom_DNN_2": "models/dnn_cnn/my_dnn_2.pkl",
    "custom_DNN_3": "models/dnn_cnn/my_dnn_3.pkl",
    "torch_DNN_1": "models/dnn_cnn/torch_dnn_1.pt",
    "torch_DNN_2": "models/dnn_cnn/torch_dnn_2.pt",
    "torch_DNN_3": "models/dnn_cnn/torch_dnn_3.pt",
    "torch_CNN_1": "models/dnn_cnn/torch_cnn_1.pt",
    "torch_CNN_2": "models/dnn_cnn/torch_cnn_2.pt",
}


@pytest.mark.parametrize("model_name,rel_path", MODEL_FILES.items(), ids=MODEL_FILES.keys())
def test_model_file_exists(model_name, rel_path):
    full_path = os.path.join(BASE_DIR, rel_path)
    assert os.path.isfile(full_path), f"Missing weight file for {model_name}: {full_path}"
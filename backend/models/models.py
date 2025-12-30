import joblib
import numpy as np
from collections import namedtuple
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import torchvision
import sys
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

#base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .base_model import BaseModel

SplitNode = namedtuple('SplitNode', ['feature', 'threshold', 'left', 'right'])
LeafNode = namedtuple('LeafNode', ['class_'])

sys.modules['__main__'].SplitNode = SplitNode
sys.modules['__main__'].LeafNode = LeafNode

models = {}

### Logistic Regression
class sklearn_logreg(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'log_reg', 'retrained_sklearn_logreg.pkl')
        self.model = joblib.load(path)
    def predict(self, X):
        return self.model.predict(X)
    
class custom_logreg(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'log_reg', 'logreg_custom_regular.npz')
        params = np.load(path)
        self.W, self.b = params['W'], params['b']
    def predict(self, X):
        def softmax(z):
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        linear = np.dot(X, self.W) + self.b
        y_pred = softmax(linear)
        return np.argmax(y_pred, axis=1)
    
models["sklearn_logreg"] = sklearn_logreg
models["custom_logreg"] = custom_logreg

### Naive Bayes
class sklearn_nb(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'naive_bayes', 'retrained_sklearn_nb.pkl')
        self.model = joblib.load(path)
    
    def predict(self, X):
        return self.model.predict(X)

class custom_nb(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'naive_bayes', 'naive_bayes_custom.npz')
        data = np.load(path, allow_pickle=True)
        self.priors = data["priors"]
        self.means = data["means"]
        self.variances = data["variances"]
        self.classes = data["classes"]
    
    def predict(self, X):
        N, D = X.shape
        K = self.priors.shape[0]
        scores = np.zeros((N, K))
        
        for k in range(K):
            prior = np.log(self.priors[k])
            mu = self.means[k]
            sigma2 = self.variances[k]
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * sigma2) + ((X - mu) ** 2) / (2 * sigma2),
                axis=1
            )
            scores[:, k] = prior + log_likelihood
        
        return self.classes[np.argmax(scores, axis=1)]

models["sklearn_nb"] = sklearn_nb
models["custom_nb"] = custom_nb

###Decision_Tree
class sklearn_DT(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'decision_tree', 'retrained_sklearn_DT.pkl')
        self.model = joblib.load(path)
    
    def predict(self, X):
        return self.model.predict(X)

class custom_DT(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'decision_tree', 'custom_DT.pkl')
        with open(path, 'rb') as f:
            self.tree = pickle.load(f)
    
    def predict_one(self, x, node):
        if isinstance(node, LeafNode):
            return node.class_
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        return self.predict_one(x, node.right)
    
    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])

models["sklearn_DT"] = sklearn_DT
models["custom_DT"] = custom_DT

### SVM
class sklearn_SVM(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'SVM', 'retrained_sklearn_SVM.pkl')
        self.model = joblib.load(path)
    
    def predict(self, X):
        return self.model.predict(X)

# Add SVM if file exists
if os.path.exists(os.path.join(BASE_DIR, 'models', 'SVM', 'retrained_sklearn_SVM.pkl')):
    models["sklearn_SVM"] = sklearn_SVM

### Random Forest
class sklearn_RF(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'random_forest', 'retrained_sklearn_RF.pkl')
        self.model = joblib.load(path)
    
    def predict(self, X):
        return self.model.predict(X)

class custom_RF(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'random_forest', 'forest.pkl')
        with open(path, 'rb') as f:
            models_state = pickle.load(f)
        
        self.trees = []
        for state in models_state:
            dt = self._DT_rf()
            dt.__dict__.update(state)
            self.trees.append(dt)
    
    class _DT_rf:
        def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.max_features = max_features
            self.tree = None
        
        def predict_one(self, x, node):
            if isinstance(node, LeafNode):
                return node.class_
            if x[node.feature] <= node.threshold:
                return self.predict_one(x, node.left)
            return self.predict_one(x, node.right)
        
        def predict(self, X):
            return np.array([self.predict_one(x, self.tree) for x in X])
    
    def predict(self, X):
        all_preds = []
        for tree in self.trees:
            preds = tree.predict(X)
            all_preds.append(preds)
        all_preds = np.array(all_preds)
        y_pred_final = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=all_preds)
        return y_pred_final

models["sklearn_RF"] = sklearn_RF
models["custom_RF"] = custom_RF

### Gradient Boost
class sklearn_GB(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'random_forest', 'retrained_sklearn_GB.pkl')
        try:
            self.model = joblib.load(path)
        except:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
    
    def predict(self, X):
        return self.model.predict(X)

class GB_classifier:
    def __init__(self, learning_rate=0.1, n_trees=100, max_depth=3):
        self.learning_rate = learning_rate
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.K = 0
        self.trees = {}
        
    def _softmax(self, preds):
        exp = np.exp(preds - np.max(preds, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predict(self, x):
        preds = np.zeros((len(x), self.K))
        for i in range(self.n_trees):
            for k in range(self.K):
                preds[:, k] += self.learning_rate * self.trees[k][i].predict(x)
        return np.argmax(preds, axis=1)

sys.modules['__main__'].GB_classifier = GB_classifier

class custom_GB(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'random_forest', 'gbdt_model.pkl')
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, X):
        return self.model.predict(X)

models["sklearn_GB"] = sklearn_GB
models["custom_GB"] = custom_GB

### KNN
class sklearn_KNN(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'KNN', 'retrained_sklearn_KNN.pkl')
        self.model = joblib.load(path)
    
    def predict(self, X):
        return self.model.predict(X)

models["sklearn_KNN"] = sklearn_KNN

class My_KNN: # classifier
    def __init__(self , k = 5):
        self.k = k
        
    def get_data(self , X , y):
        self.X_train = X
        self.y_train = y
        
    def _euclidian_distance(self , x_test):
        return np.linalg.norm(self.X_train - x_test , axis = 1) # root((a-b)^2) euclidian distance with all 

    def _find_neighbour(self , x_test):
        distances = self._euclidian_distance(x_test)
        k_nearest = np.argsort(distances)[:self.k] # select first k closest indexes
        targets = self.y_train[k_nearest]
        return np.bincount(targets.astype(int)).argmax()

    def predict(self , X_test):
        return np.array([self._find_neighbour(x) for x in X_test])

class custom_KNN(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'KNN', 'my_knn.pkl')
        with open(path , 'rb') as f:
            self.knn = pickle.load(f)
    
    def predict(self, X):
        return self.knn.predict(X);

### DNNs

class My_DNN:  # Custom class needed for unpickling
    def __init__(self, arch=None, activation="relu", optimizer="sgd", epochs=100, 
                 batch_size=32, lr=0.01, beta=0.9, rho=0.9, eps=1e-8, 
                 beta1=0.9, beta2=0.999, dropout_rate=0.0):
        self.arch = arch
        if activation == "relu":
            self._activation = self._relu
        elif activation == "sigmoid":
            self._activation = self._sigmoid
        else:
            raise ValueError("Activation must be relu or sigmoid")

        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.dropout_rate = dropout_rate
        self.params = {}
        self.cache = {}
        self.grads = {}
        self.state = {}
        self.t = 1

    def _relu(self, x, diff=False):
        if diff:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

    def _sigmoid(self, x, diff=False):
        sig = 1/(1 + np.exp(-x))
        if diff:
            return sig * (1-sig)
        return sig

    def _softmax(self, x):
        exp = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp / np.sum(exp, axis=0, keepdims=True)

    def forward(self, X):
        self.cache["A0"] = X
        for i in range(1, len(self.arch)):
            W, b = self.params["W" + str(i)], self.params["b" + str(i)]
            Z = np.dot(W, self.cache["A" + str(i - 1)]) + b
            self.cache["Z" + str(i)] = Z
            if i == len(self.arch) - 1:
                self.cache["A" + str(i)] = self._softmax(Z)
            else:
                self.cache["A" + str(i)] = self._activation(Z)
                if self.dropout_rate > 0.0:
                    D = (np.random.randn(*self.cache["A" + str(i)].shape) > self.dropout_rate).astype(float)
                    self.cache["A" + str(i)] *= D / (1 - self.dropout_rate)
                    self.cache["D" + str(i)] = D
        return self.cache["A" + str(len(self.arch) - 1)]

    def predict(self, X):
        Al = self.forward(X)
        return np.argmax(Al, axis=0)

# Register for pickle
sys.modules['__main__'].My_DNN = My_DNN

class custom_DNN_1(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'dnn_cnn', 'my_dnn_1.pkl')
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X.T)

class custom_DNN_2(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'dnn_cnn', 'my_dnn_2.pkl')
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X.T)

class custom_DNN_3(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'dnn_cnn', 'my_dnn_3.pkl')
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X.T)

models["custom_DNN_1"] = custom_DNN_1
models["custom_DNN_2"] = custom_DNN_2
models["custom_DNN_3"] = custom_DNN_3

### PyTorch DNNs
class torch_DNN_1(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'dnn_cnn', 'torch_dnn_1.pt')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.load(path, map_location=self.device, weights_only=False)
        self.model.eval()
    
    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.inference_mode():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        return preds.cpu().numpy()

class torch_DNN_2(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'dnn_cnn', 'torch_dnn_2.pt')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.load(path, map_location=self.device, weights_only=False)
        self.model.eval()
    
    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.inference_mode():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        return preds.cpu().numpy()

class torch_DNN_3(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'dnn_cnn', 'torch_dnn_3.pt')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.load(path, map_location=self.device, weights_only=False)
        self.model.eval()
    
    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.inference_mode():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        return preds.cpu().numpy()

models["torch_DNN_1"] = torch_DNN_1
models["torch_DNN_2"] = torch_DNN_2
models["torch_DNN_3"] = torch_DNN_3

### PyTorch CNN Architecture Classes (for unpickling)
class basic_conv_net1(nn.Module):
    """Basic CNN architecture - needed for unpickling torch_CNN_1"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Register for pickle
sys.modules['__main__'].basic_conv_net1 = basic_conv_net1

### PyTorch CNNs
class torch_CNN_1(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'dnn_cnn', 'torch_cnn_1.pt')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.load(path, map_location=self.device, weights_only=False)
        self.model.eval()
    
    def predict(self, X):
        transform_X = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if not isinstance(X, np.ndarray):
            raise ValueError("Input must be a NumPy array")
        if X.dtype != np.uint8:
            if X.max() <= 1.0:
                X = (X * 255).astype(np.uint8)
            else:
                X = X.astype(np.uint8)
        
        # Handle grayscale properly
        if X.ndim == 2:
            pil_image = Image.fromarray(X, mode='L')
        elif X.ndim == 3 and X.shape[2] == 1:
            pil_image = Image.fromarray(X[:, :, 0], mode='L')
        else:
            pil_image = Image.fromarray(X)
        
        tensor_image = transform_X(pil_image)
        final_X = tensor_image.unsqueeze(0).to(self.device)
        
        with torch.inference_mode():
            logits = self.model(final_X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds.cpu().numpy()

class torch_CNN_2(BaseModel):
    def __init__(self):
        path = os.path.join(BASE_DIR, 'models', 'dnn_cnn', 'torch_cnn_2.pt')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.load(path, map_location=self.device, weights_only=False)
        self.model.eval()
    
    def predict(self, X):
        from torchvision.models import EfficientNet_B0_Weights
        
        if not isinstance(X, np.ndarray):
            raise ValueError("Input must be a NumPy array")
        
        if X.ndim == 2:
            X = np.stack([X, X, X], axis=2)
        elif X.ndim == 3 and X.shape[2] == 1:
            X = np.concatenate([X, X, X], axis=2)
        
        if X.dtype != np.uint8:
            if X.max() <= 1.0:
                X = (X * 255).astype(np.uint8)
            else:
                X = X.astype(np.uint8)
        
        weights = EfficientNet_B0_Weights.DEFAULT
        auto_transforms = weights.transforms()
        pil_image = Image.fromarray(X)
        tensor_image = auto_transforms(pil_image)
        final_X = tensor_image.unsqueeze(0).to(self.device)
        
        with torch.inference_mode():
            logits = self.model(final_X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds.cpu().numpy()

models["torch_CNN_1"] = torch_CNN_1
models["torch_CNN_2"] = torch_CNN_2

print(f"Model registry initialized with {len(models)} model classes")
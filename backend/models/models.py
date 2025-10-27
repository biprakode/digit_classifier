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


SplitNode = namedtuple('SplitNode', ['feature', 'threshold', 'left', 'right'])
LeafNode = namedtuple('LeafNode', ['class_'])

models = {}

### Logistic Regression
models["sklearn_logreg"] = joblib.load('models/log_reg/logreg_sklearn.pkl')
params = np.load('models/log_reg/logreg_custom_regular.npz')
W,b = params['W'] , params['b']
def custom_logreg_predict(X):
    def softmax(z):
        exp_z = np.exp(z - np.max(z , axis = 1 , keepdims = True))
        return exp_z / np.sum(exp_z , axis = 1 , keepdims = True)
    
    linear = np.dot(X, W) + b
    y_pred = softmax(linear)
    return np.argmax(y_pred , axis = 1)
models["custom_logreg"] = custom_logreg_predict

### Naive Bayes
models["sklearn_nb"] = joblib.load('models/naive_bayes/naive_bayes_sklearn.pkl')
data = np.load('models/naive_bayes/naive_bayes_custom.npz', allow_pickle=True)
def custom_nb(X):
    priors = data["priors"]
    means = data["means"]
    variances = data["variances"]
    classes = data["classes"]
    N, D = X.shape
    K = priors.shape[0]
    scores = np.zeros((N , K))
    for k in range(K):
        prior = np.log(priors[k])
        mu = means[k]
        sigma2 = variances[k]
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + ((X - mu) ** 2) / (2 * sigma2) , axis=1)
        scores[:, k] = prior + log_likelihood
    return classes[np.argmax(scores , axis = 1)]
models["custom_nb"] = custom_nb

###Decision_Tree
models["sklearn_DT"] = joblib.load('models/decision_tree/sklearn_DT.pkl')
## def custom_DT(X): TODO

### SVM
models["sklearn_SVM"] = joblib.load('models/SVM/sklearn_SVM.pkl')
        
### Random Forest
models["sklearn_RF"] = joblib.load('models/random_forest/random_forest_sklearn.pkl')
class My_DT_rf: # sorry I HAVE to define class (younger me made dumb decisions)
    def __init__(self, max_depth=10, min_samples_split=2 , max_features = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split # min samples in a split
        self.max_features = max_features
        self.tree = None

    def gini(self , y):
        _ , counts = np.unique(y , return_counts = True)
        p = counts / len(y)
        return 1 - np.sum(p**2)
    
    def entropy(self , y):
        pk = np.unique(y , return_counts = True) / len(y)
        return np.sum(-1.*pk*np.log2(pk + 1e-9))

    def best_split(self , X , y):
        N, D = X.shape
        features = np.arange(D)
        if self.max_features is not None:
            features = np.random.choice(D, self.max_features, replace=False)
        gini_parent = self.gini(y)
        best_gain = -np.inf
        best_feature, best_threshold = None, None
        best_left_idx, best_right_idx = None, None

        for d in features:
            thresholds = np.unique(X[:, d])
            for thr in thresholds:
                left_mask = X[:, d] <= thr
                right_mask = ~left_mask
                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue
                gini_left = self.gini(y[left_mask])
                gini_right = self.gini(y[right_mask])
                gini_split = (np.sum(left_mask)/N) * gini_left + (np.sum(right_mask)/N) * gini_right
                gain = self.gini(y) - gini_split
                if gain > best_gain:
                    best_gain = gain
                    best_feature, best_threshold = d, thr
                    best_left_idx, best_right_idx = left_mask, right_mask

        return best_feature, best_threshold, best_left_idx, best_right_idx

    def build_tree(self , X , y , depth = 0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return LeafNode(class_ = np.bincount(y.astype(int)).argmax()) # return argmax class

        feature , threshold , left_idx , right_idx = self.best_split(X , y)
        if feature is None:  # No valid split
            return LeafNode(class_=np.bincount(y.astype(int)).argmax())

        left_node = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right_node = self.build_tree(X[right_idx], y[right_idx], depth + 1)
        return SplitNode(feature=feature, threshold=threshold, left=left_node, right=right_node)

    def train(self , X , y):
        self.tree = self.build_tree(X , y)

    def predict_one(self , x , node):
        if isinstance(node, LeafNode):
            return node.class_ # if leaf return leaf_class
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left) # forward left
        return self.predict_one(x, node.right) # forward right

    def predict(self , X):
        return np.array([self.predict_one(x, self.tree) for x in X])
with open('models/random_forest/forest.pkl') as f:
    models_state = pickle.load(f)
loaded_models = []
for state in models_state:
    dt = My_DT_rf()
    dt.__dict__.update(state)
    loaded_models.append(dt)
def custom_RF(X):
    all_preds = []
    for model in loaded_models:
        preds = model.predict(X)
        all_preds.append(preds)
    all_preds = np.array(all_preds)
    y_pred_final = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=all_preds)
    return y_pred_final
models['custom_RF'] = custom_RF

### Gradient Boost
models["sklearn_GB"] = joblib.load('models/random_forest/gbdt_sklearn.pkl')
class GB_classifier: # stupid young me
    def __init__(self , learning_rate = 0.1 , n_trees = 100 , max_depth = 3):
        self.learning_rate = learning_rate
        self.n_trees = n_trees
        self.max_depth = max_depth
        
    def _softmax(self , preds):
        exp = np.exp(preds)
        return exp / np.sum(exp , axis = 1 , keepdims = True)

    def train(self , X , y):
        self.K = len(np.unique(y))
        self.trees = {k: [] for k in range(self.K)} # multiclass forest of trees
        ohe_y = np.eye(self.K)[y.astype(int)]
        preds = np.zeros(ohe_y.shape)

        for i in range(self.n_trees):
            probs = self._softmax(preds)
            
            for k in range(self.K):
                # k class logitboost
                numerator = (ohe_y.T[k] - probs.T[k])
                denominator = probs.T[k] * (1 - probs.T[k])
                residuals = (self.K - 1) / self.K * numerator / denominator
                weights = denominator

                tree = DecisionTreeRegressor(criterion='friedman_mse', max_depth=self.max_depth)
                tree.fit(X , residuals , sample_weight = weights)
                self.trees[k].append(tree)
                leaf_index = tree.apply(X) # returns leaf index
                preds.T[k] += self.learning_rate * tree.predict(X)
            
    def predict(self , x):
        preds = np.zeros((len(x) , self.K))
        for i in range(self.n_trees):
            for k in range(self.K):
                preds.T[k] += self.learning_rate * self.trees[k][i].predict(x)
        return np.argmax(preds , axis = 1)

with open('models/random_forest/gbdt_model.pkl') as f:
    my_GB = pickle.load(f)
def custom_GB(X):
    y_preds = my_GB.predict(X)
    return y_preds
models["custom_GB"] = custom_GB


### KNN
models['sklearn_KNN'] = joblib.load('models/KNN/knn_sklearn.pkl')

###DNN_1


class My_DNN: #repenting for my sins
    def __init__(self, arch, activation="relu", optimizer="sgd",epochs=100, batch_size=32, lr=0.01,beta=0.9, rho=0.9, eps=1e-8, beta1=0.9, beta2=0.999 , dropout_rate=0.0):
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
        self.params = self._initialize()
        self.cache = {}
        self.grads = {}
        self._init_optimizer()

    def _relu(self , x , diff=False):
        if diff:
            return np.where(x > 0, 1, 0)
        return np.maximum(0 , x)

    def _sigmoid(self , x , diff=False):
        sig = 1/(1 + np.exp(-x))
        if diff:
            return sig * (1-sig)
        return sig

    def _softmax(self , x):
        exp = np.exp(x - np.max(x, axis=0, keepdims=True)) # stabler softmax
        return exp / np.sum(exp, axis=0, keepdims=True)

    def _initialize(self):
        params = {}
        for i in range(1, len(self.arch)):
            params["W" + str(i)] = np.random.randn(self.arch[i], self.arch[i - 1]) * np.sqrt(2. / self.arch[i - 1])
            params["b" + str(i)] = np.zeros((self.arch[i], 1))
        return params

    def _ohe(self , y):
        num_classes = self.arch[-1]
        y = y.astype(int)
        return np.eye(num_classes)[y].T

    def forward(self , X):
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
                    D = (np.random.randn(*self.cache["A" + str(i)].shape) > self.dropout_rate).astype(float) # dr mask
                    self.cache["A" + str(i)] *= D / (1 - self.dropout_rate) # scaling values (applying mask)
                    self.cache["D" + str(i)] = D

        return self.cache["A" + str(len(self.arch) - 1)]

    def _compute_cost(self , Al , y , eps = 1e-10):
        Al = np.clip(Al, eps, 1 - eps)
        m = y.shape[1]
        cost = -np.sum(y * np.log(Al)) / m
        return cost
    
    def _back_prop(self , y , output):
        m = y.shape[1]
        self.grads = {}
        L = len(self.arch) - 1

        dZ = output - y
        self.grads["W" + str(L)] = (1. / m) * np.dot(dZ, self.cache["A" + str(L - 1)].T)
        self.grads["b" + str(L)] = (1. / m) * np.sum(dZ, axis=1, keepdims=True)

        dA_prev = np.dot(self.params["W" + str(L)].T, dZ)

        for l in reversed(range(1, L)):
            if self.dropout_rate > 0.0:
                dA_prev = (dA_prev * self.cache["D" + str(l)]) / (1 - self.dropout_rate) # removing mask
        
            Z = self.cache["Z" + str(l)]
            dZ = dA_prev * self._activation(Z, diff=True)
            
            self.grads["W" + str(l)] = (1. / m) * np.dot(dZ, self.cache["A" + str(l - 1)].T)
            self.grads["b" + str(l)] = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
            
            dA_prev = np.dot(self.params["W" + str(l)].T, dZ)

    def _init_optimizer(self):
        self.state = {}
        for key in self.params:
            if self.optimizer == "momentum":
                self.state[key] = np.zeros_like(self.params[key]) # v
            elif self.optimizer == "rmsprop":
                self.state[key] = np.zeros_like(self.params[key]) # s
            elif self.optimizer == "adam":
                self.state[key] = {"m": np.zeros_like(self.params[key]), # m
                                   "v": np.zeros_like(self.params[key])} # v
        self.t = 1 #timestep

    def _update_params(self):
        if self.optimizer == "sgd":
            for key in self.params:
                self.params[key] -= self.lr * self.grads[key]

        elif self.optimizer == "momentum":
            for key in self.params:
                self.state[key] = self.beta * self.state[key] + (1 - self.beta) * self.grads[key]
                self.params[key] -= self.lr * self.state[key]

        elif self.optimizer == "rmsprop":
            for key in self.params:
                self.state[key] = self.rho * self.state[key] + (1 - self.rho) * (self.grads[key] ** 2)
                self.params[key] -= self.lr * self.grads[key] / (np.sqrt(self.state[key]) + self.eps)

        elif self.optimizer == "adam":
            for key in self.params:
                self.state[key]["m"] = self.beta1 * self.state[key]["m"] + (1 - self.beta1) * self.grads[key]
                self.state[key]["v"] = self.beta2 * self.state[key]["v"] + (1 - self.beta2) * (self.grads[key] ** 2)

                m_correct = self.state[key]["m"] / (1 - self.beta1 ** self.t)
                v_correct = self.state[key]["v"] / (1 - self.beta2 ** self.t)

                self.params[key] -= self.lr * m_correct / (np.sqrt(v_correct) + self.eps)
            self.t += 1

    def train(self, X_tr, y_tr, X_te, y_te):
        m = X_tr.shape[1]
        #flatten & apply ohe
        X_tr = X_tr.reshape(X_tr.shape[0], -1).T
        X_te = X_te.reshape(X_te.shape[0], -1).T
        if y_tr.ndim == 1:
            y_tr = self._ohe(y_tr)
        if y_te.ndim == 1:
            y_te = self._ohe(y_te)
        for epoch in range(self.epochs):
            permutation = np.random.permutation(m)
            X_tr_shuffled = X_tr[:, permutation]
            y_tr_shuffled = y_tr[:, permutation]

            for j in range(0, m, self.batch_size):
                X_batch = X_tr_shuffled[:, j:j + self.batch_size]
                y_batch = y_tr_shuffled[:, j:j + self.batch_size]

                Al = self.forward(X_batch)
                cost = self._compute_cost(Al, y_batch)
                self._back_prop(y_batch, Al)
                self._update_params()

            if (epoch + 1) % 10 == 0:
                preds = self.predict(X_tr)
                labels = np.argmax(y_tr, axis=0)
                tr_acc = np.mean(preds == labels)
                preds = self.predict(X_te)
                labels = np.argmax(y_te, axis=0)
                te_acc = np.mean(preds == labels)
                print(f"Epoch {epoch + 1}/{self.epochs}, Cost: {cost:.4f}, Train Acc: {tr_acc:.4f} ,Test Acc: {te_acc:.4f}")

    def predict(self, X):
        Al = self.forward(X)
        return np.argmax(Al, axis=0)
    
with open ('/kaggle/working/my_dnn_1.pkl' , 'rb') as f:
    my_dnn_1 = pickle.load(f)
def custom_DNN_1(X):
    preds = my_dnn_1.predict(X.T)
    return preds
models["custom_DNN_1"] = custom_DNN_1
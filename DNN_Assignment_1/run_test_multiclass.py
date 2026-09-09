import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==========================================
# 1. Load Data
# ==========================================
url = "https://raw.githubusercontent.com/noahk587/Student-Dropout-Prediction/main/data.csv"
data = pd.read_csv(url, sep=';')

# Separate features and target
X = data.drop('Target', axis=1).values
y_raw = data['Target'].values

# Encode target categories: 'Graduate', 'Dropout', 'Enrolled'
le = LabelEncoder()
y = le.fit_transform(y_raw) # 0, 1, 2
print("Class mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Helper to get one-hot encoded matrix
# ==========================================
def get_one_hot(y, num_classes=3):
    m = len(y)
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot

# ==========================================
# 2. Baseline Model (Softmax Regression)
# ==========================================
class BaselineModel:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def softmax(self, Z):
        # Z shape: (m, C)
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        num_classes = 3
        
        # One-hot encode targets for loss and gradient calculation
        y_one_hot = get_one_hot(y, num_classes)
        
        # Initialize weights of shape (n_features, num_classes)
        self.weights = np.zeros((n_features, num_classes))
        self.bias = np.zeros(num_classes)

        for i in range(self.n_iterations):
            # Forward pass
            Z = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(Z)
            
            # Loss: Categorical Cross-Entropy
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(np.sum(y_one_hot * np.log(y_pred_clipped), axis=1))
            self.loss_history.append(loss)
            
            # Gradients
            dZ = y_pred - y_one_hot
            dw = np.dot(X.T, dZ) / n_samples
            db = np.sum(dZ, axis=0) / n_samples
            
            # Parameter updates
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X):
        Z = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(Z)
        return np.argmax(y_pred, axis=1)

# ==========================================
# 3. MLP Model (Multi-class MLP)
# ==========================================
class MLP:
    def __init__(self, architecture, learning_rate=0.01, n_iterations=1000):
        self.architecture = architecture
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.parameters = {}
        self.loss_history = []
        self.cache = {}

    def initialize_parameters(self):
        np.random.seed(42)
        for l in range(1, len(self.architecture)):
            # He initialization for hidden layers
            self.parameters[f'W{l}'] = np.random.randn(self.architecture[l], self.architecture[l-1]) * np.sqrt(2.0 / self.architecture[l-1])
            self.parameters[f'b{l}'] = np.zeros((self.architecture[l], 1))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        # Column-wise softmax
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward_propagation(self, X):
        # samples as columns
        self.cache['A0'] = X.T
        A = self.cache['A0']
        L = len(self.architecture) - 1
        
        for l in range(1, L):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A) + b
            A = self.relu(Z)
            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A
            
        W_L = self.parameters[f'W{L}']
        b_L = self.parameters[f'b{L}']
        Z_L = np.dot(W_L, A) + b_L
        A_L = self.softmax(Z_L)
        self.cache[f'Z{L}'] = Z_L
        self.cache[f'A{L}'] = A_L
        
        return A_L

    def backward_propagation(self, X, y):
        m = X.shape[0]
        grads = {}
        L = len(self.architecture) - 1
        
        # Reshape y to one-hot of shape (C, m)
        y_one_hot = get_one_hot(y, self.architecture[-1]).T
        
        A_L = self.cache[f'A{L}']
        dZ = A_L - y_one_hot
        
        for l in range(L, 0, -1):
            A_prev = self.cache[f'A{l-1}']
            grads[f'dW{l}'] = np.dot(dZ, A_prev.T) / m
            grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m
            
            if l > 1:
                W = self.parameters[f'W{l}']
                Z_prev = self.cache[f'Z{l-1}']
                dZ = np.dot(W.T, dZ) * self.relu_derivative(Z_prev)
                
        return grads

    def update_parameters(self, grads):
        for l in range(1, len(self.architecture)):
            self.parameters[f'W{l}'] -= self.lr * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.lr * grads[f'db{l}']

    def compute_loss(self, y_pred, y_true):
        # y_pred: (C, m), y_true: (m,)
        m = len(y_true)
        y_one_hot = get_one_hot(y_true, self.architecture[-1]).T
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_one_hot * np.log(y_pred_clipped), axis=0))
        return loss

    def fit(self, X, y):
        self.initialize_parameters()
        for i in range(self.n_iterations):
            y_pred = self.forward_propagation(X)
            loss = self.compute_loss(y_pred, y)
            self.loss_history.append(loss)
            grads = self.backward_propagation(X, y)
            self.update_parameters(grads)
        return self

    def predict(self, X):
        y_pred = self.forward_propagation(X)
        predictions = np.argmax(y_pred, axis=0)
        return predictions

# ==========================================
# 4. Metrics Calculation (Macro-averaged)
# ==========================================
def calculate_metrics(y_true, y_pred):
    classes = np.unique(y_true)
    num_classes = len(classes)
    
    tp_total = 0
    precision_sum = 0
    recall_total_sum = 0
    f1_total_sum = 0
    
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        precision_sum += p
        recall_total_sum += r
        f1_total_sum += f
        tp_total += tp
        
    accuracy = tp_total / len(y_true)
    precision = precision_sum / num_classes
    recall = recall_total_sum / num_classes
    f1 = f1_total_sum / num_classes
    
    return accuracy, precision, recall, f1

# ==========================================
# 5. Run Training
# ==========================================
# Train Baseline
baseline_start = time.time()
baseline_model = BaselineModel(learning_rate=0.1, n_iterations=1000)
baseline_model.fit(X_train_scaled, y_train)
baseline_predictions = baseline_model.predict(X_test_scaled)
baseline_training_time = time.time() - baseline_start

# Train MLP
mlp_start = time.time()
# Output layer has size 3 for 3 classes
mlp_architecture = [36, 16, 8, 3]
mlp_model = MLP(architecture=mlp_architecture, learning_rate=0.05, n_iterations=1000)
mlp_model.fit(X_train_scaled, y_train)
mlp_predictions = mlp_model.predict(X_test_scaled)
mlp_training_time = time.time() - mlp_start

baseline_metrics = calculate_metrics(y_test, baseline_predictions)
mlp_metrics = calculate_metrics(y_test, mlp_predictions)

print("BASELINE RESULTS:")
print(f"Initial loss: {baseline_model.loss_history[0]:.4f}")
print(f"Final loss: {baseline_model.loss_history[-1]:.4f}")
print(f"Accuracy: {baseline_metrics[0]:.4f}")
print(f"Precision: {baseline_metrics[1]:.4f}")
print(f"Recall: {baseline_metrics[2]:.4f}")
print(f"F1 Score: {baseline_metrics[3]:.4f}")
print(f"Training Time: {baseline_training_time:.4f}s")

print("\nMLP RESULTS:")
print(f"Initial loss: {mlp_model.loss_history[0]:.4f}")
print(f"Final loss: {mlp_model.loss_history[-1]:.4f}")
print(f"Accuracy: {mlp_metrics[0]:.4f}")
print(f"Precision: {mlp_metrics[1]:.4f}")
print(f"Recall: {mlp_metrics[2]:.4f}")
print(f"F1 Score: {mlp_metrics[3]:.4f}")
print(f"Training Time: {mlp_training_time:.4f}s")

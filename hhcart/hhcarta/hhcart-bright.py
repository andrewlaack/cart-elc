import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def compute_householder_matrix(d):
    p = d.shape[0]
    e1 = np.zeros_like(d)
    e1[0] = 1.0
    u = e1 - d
    u = u / np.linalg.norm(u)
    H = np.eye(p) - 2 * np.outer(u, u)
    return H

def reflect_dataset(X, d):
    H = compute_householder_matrix(d)
    X_reflected = X @ H
    return X_reflected, H

def compute_twoing_score(y_left, y_right):
    # Avoid degenerate splits.
    if len(y_left) == 0 or len(y_right) == 0:
        return 0

    total = len(y_left) + len(y_right)
    P_L = len(y_left) / total
    P_R = len(y_right) / total

    classes = np.unique(np.concatenate([y_left, y_right]))
    sum_diff = 0
    for cls in classes:
        P_Li = np.sum(y_left == cls) / len(y_left)
        P_Ri = np.sum(y_right == cls) / len(y_right)
        sum_diff += np.abs(P_Li - P_Ri)

    return P_L * P_R * (sum_diff ** 2)

def find_best_reflection(X, y):
    """
    For each class in y, compute all eigenvectors (from the class covariance),
    reflect the dataset for each eigenvector candidate, and then for every axis in 
    the reflected feature space try all axis-aligned splits using the unique sample 
    values (without interpolation) to evaluate the twoing score.
    Returns the parameters of the best split found.
    """
    best_score = -np.inf
    best_params = None

    for cls in np.unique(y):
        class_indices = np.where(y == cls)[0]
        if len(class_indices) < 2:
            continue
        class_data = X[class_indices]
        cov = np.cov(class_data, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Evaluate all eigenvector candidates (each column in eigvecs)
        for i in range(eigvecs.shape[1]):
            d_candidate = eigvecs[:, i]
            X_reflected, H = reflect_dataset(X, d_candidate)
            p = X_reflected.shape[1]
            # For each axis in the reflected space, check axis-aligned splits.
            for j in range(p):
                values = X_reflected[:, j]
                unique_vals = np.unique(values)
                if len(unique_vals) < 2:
                    continue
                # Use each unique value directly as a candidate threshold.
                for threshold in unique_vals:
                    left_idx = values <= threshold
                    right_idx = ~left_idx
                    if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                        continue
                    score = compute_twoing_score(y[left_idx], y[right_idx])
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'H': H,
                            'd_candidate': d_candidate,  # Store the eigenvector used.
                            'axis': j,
                            'threshold': threshold,
                            'left_idx': left_idx,
                            'right_idx': right_idx,
                            'score': score,
                            'class_used': cls,
                            'X_reflected': X_reflected
                        }
    return best_params

class Node:
    def __init__(self, is_leaf=False, prediction=None, H=None, d_candidate=None, axis=None,
                 threshold=None, score=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.prediction = prediction      # Majority class for leaf nodes.
        self.H = H                        # Reflection matrix used at the node.
        self.d_candidate = d_candidate    # Eigenvector used for reflection.
        self.axis = axis                  # Axis in the reflected space used for splitting.
        self.threshold = threshold        # Threshold for the axis-aligned split.
        self.score = score                # Twoing score for the chosen split.
        self.left = left                  # Left child node.
        self.right = right                # Right child node.

class HHCartClassifier:
    def __init__(self, max_depth=4, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        
    def _build_tree(self, X, y, depth):
        # Stopping criteria: max depth reached, too few samples, or pure node.
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.unique(y).size == 1:
            majority_class = np.bincount(y.astype(int)).argmax()
            return Node(is_leaf=True, prediction=majority_class)
        
        best_params = find_best_reflection(X, y)
        if best_params is None:
            majority_class = np.bincount(y.astype(int)).argmax()
            return Node(is_leaf=True, prediction=majority_class)

        H = best_params['H']
        d_candidate = best_params['d_candidate']
        axis = best_params['axis']
        threshold = best_params['threshold']
        left_idx = best_params['left_idx']
        right_idx = best_params['right_idx']
        score = best_params['score']

        # If the split does not actually separate the data, create a leaf.
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            majority_class = np.bincount(y.astype(int)).argmax()
            return Node(is_leaf=True, prediction=majority_class)
        
        # Build left and right subtrees.
        left_node = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_node = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return Node(is_leaf=False, H=H, d_candidate=d_candidate, axis=axis, threshold=threshold, score=score,
                    left=left_node, right=right_node)

    def _predict_sample(self, x, node):
        if node.is_leaf:
            return node.prediction
        # Reflect the sample using the node's reflection matrix.
        x_reflected = x @ node.H
        # Use the stored axis in the reflected space for the split.
        value = x_reflected[node.axis]
        if value <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

def count_leaves(node):
    """Recursively count the number of leaf nodes in the tree."""
    if node.is_leaf:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Load dataset.
df = pd.read_csv('../../datasets/bright.csv')
y = df[df.columns[-1]]
df = df.drop(df.columns[-1], axis=1)
X = df.to_numpy(dtype=np.float64)

# Define hyperparameters
depths = range(1, 6)
n_splits = 5
n_trials = 10
seed = 15

# Store results: one list per depth collecting the mean accuracy and tree size from each iteration
results = {depth: [] for depth in depths}
tree_sizes = {depth: [] for depth in depths}

# For each iteration, generate the folds once and use them for each depth.
for trial in range(n_trials):
    print("TRIAL: " + str(trial + 1))
    # Create KFold splits for this trial
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=(trial + seed))
    folds = list(kf.split(X))
    
    for depth in depths:
        print("DEPTH: " + str(depth))
        fold_accuracies = []
        fold_sizes = []
        
        for train_index, test_index in folds:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Create and train the classifier with the current depth
            clf = HHCartClassifier(max_depth=depth, min_samples_split=2)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            
            # Compute and store the accuracy and tree size (number of leaf nodes)
            fold_accuracies.append(accuracy_score(y_test, preds))
            fold_sizes.append(count_leaves(clf.root))
        
        # Store the mean accuracy and mean tree size from the folds of this trial
        results[depth].append(np.mean(fold_accuracies))
        tree_sizes[depth].append(np.mean(fold_sizes))

# Write the final averaged results over all iterations to file
with open("resultsBright.txt", "w") as f:
    f.write("Grid Search Results:\n")
    for depth in depths:
        avg_accuracy = np.mean(results[depth]) * 100 # percent
        std_accuracy = np.std(results[depth]) * 100 # percent
        avg_size = np.mean(tree_sizes[depth])
        std_size = np.std(tree_sizes[depth])
        f.write(f"Depth: {depth} Avg Accuracy: {avg_accuracy:.1f} (Std: {std_accuracy:.1f}), ")
        f.write(f"Avg # of Leaves: {avg_size:.1f} (Std: {std_size:.1f})\n")

import numpy as np
import pandas as pd
import decision_tree
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

df = pd.read_csv('../../datasets/diabetes.csv')
y = df['Outcome']
df = df.drop(df.columns[-1], axis=1)
X = df.to_numpy(dtype=np.float64)

# Define hyperparameters
depths = [1,2,3,4,5]
criteria = ["gini", "twoing", "information gain"]
lcs = [1,2]
n_splits = 5
n_trials = 10
seed = 71

# Store results
accuracies = {lc: {depth: [] for depth in depths} for lc in lcs}
treeSizes = {lc: {depth: [] for depth in depths} for lc in lcs}
foldSizes = []
foldAccuracies = []

# Store results
results = {criterion: {lc: {depth: [] for depth in depths} for lc in lcs} for criterion in criteria}
tree_sizes = {criterion: {lc: {depth: [] for depth in depths} for lc in lcs} for criterion in criteria}

for trial in range(n_trials):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=(trial + seed))
    
    for criterion in criteria:
        for lc in lcs:
            for depth in depths:
                fold_accuracies = []
                fold_sizes = []
                
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    clf = decision_tree.ELCClassifier(depth, lc, 100, criterion)
                    clf.fit(X_train.ravel(), y_train.shape[0], y_train, int(X_train.size / y_train.shape[0]))
                    preds = clf.predict(X_test.ravel(), y_test.shape[0], int(X_test.size / y_test.shape[0]))
                    
                    fold_accuracies.append(100 * accuracy_score(y_pred=preds, y_true=y_test))
                    fold_sizes.append(clf.getSplits() + 1) # leaves not splits
                
                results[criterion][lc][depth].append(np.mean(fold_accuracies))
                tree_sizes[criterion][lc][depth].append(np.mean(fold_sizes))
                                                                                 
with open("resultsDiabetes.txt", "w") as f:
    f.write("Results:\n")
    for criterion in criteria:
        f.write("\n")
        f.write(f"{criterion}:\n")
        for lc in lcs:
            for depth in depths:
                avg_accuracy = np.mean(results[criterion][lc][depth])
                std_accuracy = np.std(results[criterion][lc][depth])
                avg_size = np.mean(tree_sizes[criterion][lc][depth])
                std_size = np.std(tree_sizes[criterion][lc][depth])
                
                f.write(f"LCs: {lc}, Depth: {depth} Avg Accuracy: {avg_accuracy:.1f} (Std: {std_accuracy:.1f})")
                f.write(f", Avg # of Leaves: {avg_size:.1f} (Std: {std_size:.1f})\n")

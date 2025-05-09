import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import os

# Create output directory
os.makedirs("images", exist_ok=True)

# Dark mode settings using matplotlib only
# plt.rcParams.update({
#     "axes.facecolor": "#000000",
#     "axes.edgecolor": "#333333",
#     "figure.facecolor": "#000000",
#     "savefig.facecolor": "#000000",
#     "text.color": "white",
#     "axes.labelcolor": "white",
#     "xtick.color": "white",
#     "ytick.color": "white",
#     "grid.color": "gray",
#     "axes.grid": True
# })

# Load dataset
df = pd.read_csv('./diabetes.csv')

# Extract features and target
X_bmi = df['BMI'].to_numpy()
X_glucose = df['Glucose'].to_numpy()
y = df['Outcome'].to_numpy()

# Combine features into a 2D array
data = np.column_stack((X_bmi, X_glucose))

# Create meshgrid for decision region plotting
xx, yy = np.meshgrid(np.linspace(0, 70, 500), np.linspace(0, 200, 500))
grid = np.c_[xx.ravel(), yy.ravel()]

# Plot the original data
plt.figure(figsize=(12, 6))
plt.scatter(X_bmi, X_glucose, c=y, cmap='bwr', s=60, edgecolors="#000000")
plt.xlabel("BMI")
plt.ylabel("Glucose")
plt.title("Diabetes Dataset: BMI vs Glucose")
plt.xlim(0, 70)
plt.ylim(0, 200)
plt.tight_layout()
plt.savefig("original_diabetes_plot_bmi_glucose.png", dpi=300, bbox_inches='tight')
plt.close()

# Train decision trees and plot decision boundaries
for depth in range(1, 5):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(data, y)
    preds = clf.predict(data)

    # Plot decision tree at max depth
    if depth == 4:
        fig, ax = plt.subplots(figsize=(14, 14))
        plot_tree(clf, feature_names=["BMI", "Glucose"], class_names=["No Diabetes", "Diabetes"], ax=ax, filled=True)
        
        # Save decision tree plot with a black background
        plt.savefig("tree2.pdf", bbox_inches='tight')
        plt.close()

    # Print accuracy
    acc = accuracy_score(y_pred=preds, y_true=y)
    print(f"DEPTH: {depth} - Accuracy: {acc:.4f}")

    # Plot decision boundaries
    Z = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

    plt.figure(figsize=(12, 6))
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#0000FF', '#FF0000'], alpha=0.3)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

    plt.scatter(X_bmi, X_glucose, c=y, cmap='bwr', s=60, edgecolors="#000000")
    plt.xlabel("BMI")
    plt.ylabel("Glucose")
    plt.title(f"Decision Tree (Depth {depth})")
    plt.xlim(0, 70)
    plt.ylim(0, 200)
    plt.tight_layout()
    plt.savefig(f"images/diabetes_tree_bmi_glucose_depth_{depth}.png", dpi=300, bbox_inches='tight')
    plt.close()

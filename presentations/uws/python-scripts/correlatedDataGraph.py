import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import os

# Create directory for output images if not exists
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

ARRSIZE = 500

# Generate data for two classes
X_red = np.random.rand(ARRSIZE) * 10
X_blue = np.random.rand(ARRSIZE) * 10

y_red = X_red / 1.5 + np.random.normal(0, 0.5, ARRSIZE)
y_blue = X_blue / 1.5 + np.random.normal(0, 0.5, ARRSIZE)
y_blue += 1.5  # Shift blue class upward

# Combine the data
X_combined = np.concatenate([X_red, X_blue])
y_combined = np.concatenate([y_red, y_blue])
labelsRed = np.zeros(ARRSIZE)
labelsBlue = np.ones(ARRSIZE)
labels = np.concatenate([labelsRed, labelsBlue])

# Prepare the feature matrix for training
data = np.column_stack((X_combined, y_combined))

# Create meshgrid for decision boundaries
xx, yy = np.meshgrid(np.linspace(0, 10, 500), np.linspace(0, 10, 500))
grid = np.c_[xx.ravel(), yy.ravel()]

# Save the original scatter plot
plt.figure(figsize=(16, 9))
plt.scatter(X_combined, y_combined, c=labels, cmap='bwr', s=60, edgecolors='#000000')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel('Tumor Size')
plt.ylabel('Mass')
plt.tight_layout()
plt.savefig("original_graph.png", dpi=300, bbox_inches='tight')
plt.close()

# Loop through depths
for depth in range(1, 21):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(data, labels)

    Z = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

    plt.figure(figsize=(16, 9))

    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#0000FF', '#FF0000'], alpha=0.3)
    plt.contour(xx, yy, Z, levels=[0.5], colors='#000000', linewidths=2)

    plt.scatter(X_combined, y_combined, c=labels, cmap='bwr', s=60, edgecolors='#000000')

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel('Tumor Size')
    plt.ylabel('Mass')
    plt.tight_layout()
    plt.savefig(f"images/decision_tree_depth_{depth}.png", dpi=300, bbox_inches='tight')
    plt.close()

    if depth == 20:
        fig, ax = plt.subplots(figsize=(90, 90))
        
        # Set dark background

        # Plot the tree without default fill
        plot_tree(
            clf,
            ax=ax,
            filled=True,
            feature_names=["Tumor Size", "Mass"],
            class_names=["Red", "Blue"],
            impurity=False,
            proportion=False,
            rounded=True,
            fontsize=10  # Use readable size, adjust as needed
        )

        # Save with dark facecolor
        plt.savefig("tree3.pdf", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()

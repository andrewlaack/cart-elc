import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn import metrics

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

df = pd.read_csv('./diabetes.csv')
X = df['BMI']
y = df['Glucose']
color = df['Outcome']

xSub = []
ySub = []
X = X.to_numpy()
y = y.to_numpy()

# Create scatter plot with dark styling
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c=color, cmap='bwr', s=60, edgecolors="#000000")
plt.xlabel("BMI")
plt.ylabel("Glucose")
plt.xlim([0, 50])
plt.tight_layout()

# Save the graph as dia4.pdf
plt.savefig("dia4.pdf")
plt.close()

# Decision Tree for prediction
targ = []

for i in range(0, len(X)):
    targ.append([X[i], y[i]])

dt = tree.DecisionTreeClassifier(max_depth=5)
dt.fit(targ, color)
preds = dt.predict(targ)

# Print the accuracy score
print(metrics.accuracy_score(y_pred=preds, y_true=color))

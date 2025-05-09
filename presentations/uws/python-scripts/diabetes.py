import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree


df = pd.read_csv('./diabetes.csv')
X = df['BMI']
y = df['Outcome']

xSub = []
ySub = []
X = X.to_numpy()
y = y.to_numpy()

for i in range(60, 90):
    if y[i] == 0:
        if X[i] > 32:
            continue
    xSub.append(X[i])
    ySub.append(y[i])

xSub = np.array(xSub).reshape(-1, 1)
ySub = np.array(ySub).reshape(-1, 1)

clf = DecisionTreeClassifier(max_depth=1)
clf.fit(xSub, ySub)

# Save decision tree with dark background
fig, ax = plt.subplots(figsize=(8, 6))
plot_tree(clf, filled=True, feature_names=["BMI"], class_names=["No Diabetes", "Diabetes"], ax=ax)
plt.tight_layout()
plt.savefig("tree1.pdf")
plt.close()

# Save first scatter plot (dia1)
plt.figure(figsize=(10, 6))
plt.scatter(xSub, ySub, c=ySub, cmap='bwr', s=60, edgecolors='#000000')
plt.xlabel("BMI")
plt.ylabel("Diagnosis")
plt.xlim([0, 50])
plt.tight_layout()
plt.savefig("dia1.pdf")
plt.close()

# Save second scatter plot with vertical line at 32.25 (dia2)
plt.figure(figsize=(10, 6))
plt.scatter(xSub, ySub, c=ySub, cmap='bwr', s=60, edgecolors="#000000")

# Add shaded regions
plt.axvspan(0, 32.25, color='blue', alpha=0.3)  # Light blue to the left
plt.axvspan(32.25, 50, color='red', alpha=0.3)  # Light red to the right

# Vertical decision boundary
plt.axvline(32.25, color="black", linestyle="-")

plt.xlabel("BMI")
plt.ylabel("Diagnosis")
plt.xlim([0, 50])
plt.tight_layout()
plt.savefig("dia2.pdf")
plt.close()

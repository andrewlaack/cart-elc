import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
X = df['BMI'].to_numpy()
y = df['Outcome'].to_numpy()

# Create a scatter plot using matplotlib only
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c=y, cmap='bwr', s=40, edgecolors='#000000')

plt.xlabel("BMI")
plt.ylabel("Diagnosis")

plt.axvspan(0, 32.25, color='blue', alpha=0.3)  # Light blue to the left
plt.axvspan(32.25, 50, color='red', alpha=0.3)  # Light red to the right

plt.axvline(32.25, color="black", linestyle="-")


plt.xlim([0, 50])
plt.tight_layout()

# Save the figure instead of displaying it
plt.savefig("dia3.pdf")

plt.close()  # Closes the plot so it doesn't display or use memory

# Calculate classification accuracy based on threshold
wrong = 0
for i in range(len(X)):
    if (y[i] == 1 and X[i] < 32.25) or (y[i] == 0 and X[i] > 32.25):
        wrong += 1

print((len(X) - wrong) / len(X))

import matplotlib.pyplot as plt
import seaborn as sns
import random

x = []
y = []
target = []

for i in range(0,1000):
    rnd = (random.random() * 300) + 100 
    rndAdd = rnd + (random.random() * 140 - 70)
    x.append(rndAdd)
    height = (random.random() * 40 + 30) + rndAdd * 0.1
    y.append(height)

    if rnd / height > 4:
        target.append(1)
    else:
        target.append(0)
plt.scatter(y,x, c=target, cmap='bwr')

plt.show()

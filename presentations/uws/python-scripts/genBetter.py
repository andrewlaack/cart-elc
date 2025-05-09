import matplotlib.pyplot as plt
import seaborn as sns
import random

x = []
y = []
for i in range(0,100):
    rnd = (random.random() * 300) + 100 
    rndAdd = rnd + (random.random() * 140 - 70)

    x.append(rndAdd)

    if rnd > (286-249)/2 + 249:
        y.append(1)
    else:
        y.append(0)
plt.scatter(x,y, c=y, cmap='bwr')
plt.axvline((286-249)/2 + 249)

# plt.show()

wrong = 0
for i in range(0,len(x)):
    if x[i] > (286-249)/2 + 249:
        if y[i] == 0:
            wrong += 1
    else:
        if y[i] == 1:
            wrong += 1
print((len(x) - wrong) / len(x))

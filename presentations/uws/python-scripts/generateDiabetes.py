import matplotlib.pyplot as plt
import seaborn as sns

x = [249 , 286 , 380 , 112 , 385 , 163 , 208 , 121 , 166 , 218]
y = [0 ,1 ,1 ,0 ,1 ,0 ,0 , 0 ,0 ,0]

plt.scatter(x,y, c=y, cmap='bwr')
plt.show()


x = [249 , 286 , 380 , 112 , 385 , 163 , 208 , 121 , 166 , 218]
y = [0 ,1 ,1 ,0 ,1 ,0 ,0 , 0 ,0 ,0]

plt.scatter(x,y, c=y, cmap='bwr')
plt.axvline((286-249)/2 + 249)

plt.show()

import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 3, 4, 5]
y = [12, 22, 15, 11, 18]
plt.plot(x, y, color='red')
plt.grid(color='0.9')
plt.show()


x1 = np.arange(-1, 6)
y1 = 1/2 * x1 + 1/2
y2 = -2 * x1 + 7
plt.plot(x1, y1)
plt.plot(x1, y2)
plt.axis('equal')
plt.grid()
plt.show()


values = [20, 59, 70]
labels = ['A', 'B', 'C']
plt.bar(labels, values)
plt.show()


plt.rcParams['font.sans-serif']=['SimHei']  #顯示中文
plt.plot([3,7,9],[1,5,10],marker='*', markersize=10, color='#f19790',label='星星')
plt.legend(loc='upper left')
for i in ['top','right']:
    plt.gca().spines[i].set_visible(False)

plt.title("Title")
plt.ylabel("AAA")
plt.xlabel("BBB") 
plt.show()


ironman_hist = np.random.randn(100)
plt.hist(ironman_hist, bins=10, color='#f19790', edgecolor="LightSteelBlue")
plt.show()


th = np.arange(0, 360)
x3 = np.cos(np.radians(th))
y3 = np.sin(np.radians(th))
plt.plot(x3, y3)
plt.axis('equal')
plt.grid()
plt.show()
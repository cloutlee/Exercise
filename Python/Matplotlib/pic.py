import matplotlib.pyplot as plt
import numpy as np

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



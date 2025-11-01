import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = np.random.randn(50).cumsum()
sns.histplot(data=data)
plt.show()
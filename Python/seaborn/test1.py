import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# data = np.random.randn(50).cumsum()
# sns.histplot(data=data)
# plt.show()

x = np.linspace(1, 7, 50)
y = 3 + 2 * x + 1.5 * np.random.randn(len(x))
df = pd.DataFrame({'xData': x, 'yData': y}) 
sns.regplot(x='xData', y='yData', data=df)
plt.show()
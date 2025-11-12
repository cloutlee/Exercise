import pandas as pd
import numpy as np


# -----------Series-----------

data = [1, 2, 3]
data1 = {'a': 1, 'b': 2, 'c': 3}
data2 = (1, 2, 3)
data3 = np.array([1, 2, 3])
s = pd.Series(data)
s1 = pd.Series(data1)
s2 = pd.Series(data2)
s3 = pd.Series(data3)
# print(s)
# print(s1)
# print(s2)
# print(s3)
# print(s[1:3])
# print(s[0])
# print(s.describe())
# print(s.sum())
# print(s * 2)



# -----------DataFrame-----------

data = {'Name': ['Tom', 'Nick', 'Krish', 'Jack'],
        'Age': [20, 21, 19, 18],
        'City': ['NY', 'LA', 'Chicago', 'Houston']}
df = pd.DataFrame(data)
# print(df)
# print(df.head(2))
# print(df['Name'])
# print(df.loc[0])
# print(df.iloc[0, 1])
# print(df.describe())
# print(df['Age'].mean())
# print(df.sort_values(by='Age'))

d1 = pd.read_csv('C:\\Users\\clout\\Desktop\\creditcard\\creditcard.csv')
# print(d1.head())      #顯示前5筆
# print(d1.tail(3))     #顯示後3筆
# print(d1.info())      #顯示資料摘要



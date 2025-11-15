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

data4 = {'Name': ['Tom', 'Nick', None, 'Jack'],
        'Age': [20, 21, 19, None],
        'City': ['NY', 'LA', 'Chicago', 'Houston']}
df1 = pd.DataFrame(data4)
# print(df)
# print(df.head(2))
# print(df['Name'])
# print(df.loc[0])
# print(df.iloc[0, 1])
# print(df.describe())
# print(df['Age'].mean())
# print(df.sort_values(by='Age'))
# print(df.dropna())  # 刪除有缺失值的列
# print(df.fillna({'Name': '???', 'Age': df['Age'].mean()}))  # 填補缺失值

data5 = {'A': [1, 2, 3],
         'B': [4, 5, 6]}

data6 = {'A': [1, 8, 9],
         'B': [10, 11, 12]}

df2 = pd.DataFrame(data5)
df3 = pd.DataFrame(data6)
# print(pd.concat([df2, df3], ignore_index=True)) # 合併兩個DataFrame

df4 = pd.DataFrame(data5, index=['row1', 'row2', 'row3'])
# print(df4.loc['row1'])
# print(df4.loc[:, 'A'])
# print(df4.iloc[0])
# print(df4.iloc[:, 0])
# print(df4.at['row1', 'A'])
# print(df4.iat[0, 0])
# print(df4.query('A > 1'))
# print(df4.where(df4 > 4))




d1 = pd.read_csv('C:\\Users\\clout\\Desktop\\creditcard\\creditcard.csv')
# print(d1.head())      #顯示前5筆
# print(d1.tail(3))     #顯示後3筆
# print(d1.info())      #顯示資料摘要



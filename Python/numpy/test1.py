import numpy as np

tmplist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
arr = np.array(tmplist)

# print(arr)
# print(arr.shape)
# print(arr.dtype)
# print(type(arr))
# print(arr.ndim)
# print(arr.size)

# print(arr[:])       #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# print(arr[1:3])     #[2, 3]
# print(arr[2:])      #[3, 4, 5, 6, 7, 8, 9, 10]
# print(arr[:3])      #[1, 2, 3]      
# print(arr[1:8:2])   #start:end:step [2, 4, 6, 8]
# print(arr[-1])      #10
# print(arr[-2])      #9
# print(arr[-3:])     #[8, 9, 10]
# print(arr[:-3])     #[1, 2, 3, 4, 5, 6, 7]
# print(arr[(arr >= 3) & (arr < 9)])    #[3 4 5 6 7 8]
# print(arr[arr % 2 == 0])              #[ 2  4  6  8 10]

# print(np.zeros(3))          #[0. 0. 0.]
# print(np.zeros((2,3)))      #[[0. 0. 0.]
#                             # [0. 0. 0.]]
# print(np.ones(3))           #[1. 1. 1.]
# print(np.arange(1, 10, 2))  #[1 3 5 7 9]建立等差數列(start, end, step)
# print(np.linspace(0, 1, 5)) #[0.   0.25 0.5  0.75 1. ]建立等差數列(start, end, number of elements)
# print(np.random.random(3))  #建立3個0~1之間的隨機數

# arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# arr_3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# print(arr_2d.shape)     #(3, 3)
# print(arr_3d.shape)     #(2, 2, 3)
# print(arr_2d[range(3), range(3)])   #[1 5 9] 取對角線元素
# print(arr_2d[:, 0])     #[1 4 7] 取第一欄

# print(np.sum(arr))          #求和
# print(np.mean(arr))         #平均值
# print(np.std(arr))          #標準差
# print(np.var(arr))          #變異數
# print(np.min(arr))          #最小值
# print(np.max(arr))          #最大值
# print(np.exp(arr))          #指數
# print(np.sqrt(arr))         #平方根
# print(np.log(arr))          #自然對數
# print(np.sin(arr))          #正弦值
# print(np.cos(arr))          #餘弦值
# print(np.tan(arr))          #正切值

# print(np.sort(arr))           #排序
# print(np.argmax(arr))         #最大值索引
# print(np.argmin(arr))         #最小值索引

# newarr = arr.reshape(2, 5)
# print(newarr)
# print(newarr.T)
# print(newarr.flatten())



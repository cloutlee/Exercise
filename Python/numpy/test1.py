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

# print(np.zeros(3))          #[0. 0. 0.]
# print(np.zeros((2,3)))      #[[0. 0. 0.]
#                             # [0. 0. 0.]]
# print(np.ones(3))           #[1. 1. 1.]
# print(np.arange(1, 10, 2))  #[1 3 5 7 9]建立等差數列(start, end, step)
# print(np.linspace(0, 1, 5)) #[0.   0.25 0.5  0.75 1. ]建立等差數列(start, end, number of elements)

# arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# arr_3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# print(arr_2d.shape)     #(3, 3)
# print(arr_3d.shape)     #(2, 2, 3)
# print(arr_2d[range(3), range(3)])   #[1 5 9] 取對角線元素
# print(arr_2d[:, 0])     #[1 4 7] 取第一欄


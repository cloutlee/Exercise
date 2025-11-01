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

print(np.zeros(3))        #[0. 0. 0.]
print(np.zeros((2,3)))    #[[0. 0. 0.]
                          # [0. 0. 0.]]
print(np.ones(3))         #[1. 1. 1.]



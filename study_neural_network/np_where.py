import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

result = np.where(arr > 5)
print(f"result: {result}")
print(result[0])
print(result[0].dtype)
print(result[1])
print(result[1].dtype)
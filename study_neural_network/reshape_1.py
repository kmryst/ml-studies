import numpy as np

# 元のデータ
data = np.array([[1,2], [3,4], [5,6]])
print("元のデータ shape", data.shape)
print(data)
print()

# 異なるshapeに変形
shape_3_1_2 = data.reshape(3,1,2)
shape_1_3_2 = data.reshape(1,3,2)

print("shape(3,1,2):")
print(shape_3_1_2)
print("shape:", shape_3_1_2.shape)
print()

print("shape(1,3,2):")
print(shape_1_3_2)
print("shape:", shape_1_3_2.shape)

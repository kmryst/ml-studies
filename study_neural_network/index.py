import numpy as np

# 5個の2次元点を生成
x = np.random.randn(5, 2)

print("元の配列:")
print(x)

# 1個目の点（インデックス0）のみを移動
print(x[0])
print(x[0][1])
x[0] += np.array([2, 2])

print("\n1個目の点を移動後:")
print(x)
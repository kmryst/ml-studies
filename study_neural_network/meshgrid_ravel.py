import numpy as np

# meshgridを使って2次元グリッドを作成
x = np.array([1, 2, 3])
y = np.array([4, 5])
X, Y = np.meshgrid(x, y)

print("X:")
print(X)
print("\nY:")
print(Y)

# ravel()を使って平坦化
X_flat = X.ravel()
Y_flat = Y.ravel()

print("\n平坦化されたX:")
print(X_flat)
print("\n平坦化されたY:")
print(Y_flat)

# 平坦化されたXとYを組み合わせて座標のリストを作成
coords = np.array([X_flat, Y_flat])
coords_T = np.array([X_flat, Y_flat]).T
print("\n座標のリスト:")
print(coords)
print("\n座標のリスト:")
print(coords_T)
print()

coords_3D = coords_T.reshape(Y.shape[0], Y.shape[1], 2)
print(coords_T.shape)
print(X.shape)
print(Y.shape)
print()
print(coords_3D)

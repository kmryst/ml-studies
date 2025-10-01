
import numpy as np
import matplotlib.pyplot as plt

# データの生成
np.random.seed(42)
n_points = 5
# n_points = 100


# 2つのクラスのデータを生成
x0 = np.random.randn(n_points, 2) # random normal distribution
x1 = np.random.randn(n_points, 2) + np.array([2, 2])

# データとラベルの結合
x = np.vstack((x0, x1)) # vertical(垂直) stack(積み重ねる)
print(f"x: \n {x}")
y = np.hstack((np.zeros(n_points), np.ones(n_points))) # horizontal(水平)
print(f"y: \n {y}")
z= np.hstack((x0, x1))
print(f"z: \n {z}")


# 散布図の作成
plt.figure(figsize=(10, 8)) # 10インチ × 8インチ

for cl in [0, 1]:
    plt.scatter(x=x[y == cl, 0],
                y=x[y == cl, 1],
                c=['blue', 'red'][cl], # インデックス参照 clを色と紐づけた
                label=f'Class {cl}') # これだけじゃ凡例は表示されない
    print(f"cl = {cl}: \n y == cl: \n {y == cl}")
    print(f"cl = {cl}: \n x: \n {x[y == cl, 0]} \n y: \n {x[y == cl, 1]}")

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter plot by class')
plt.legend() # 凡例
plt.grid(True)

plt.show()
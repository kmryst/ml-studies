import numpy as np
from sklearn.datasets import load_iris


# irisデータセットを読み込む
iris = load_iris()
X_data = iris.data
y_data = iris.target

print(f"x_data: \n {X_data}")
print(f"y_data: \n {y_data}")
print(f"y_data == 0: \n {(y_data == 0)}")

# 各クラスから3行ずつ取り出す
class_0 = np.where(y_data == 0)[0][:3]
class_1 = np.where(y_data == 1)[0][:3]
class_2 = np.where(y_data == 2)[0][:3]


print(f"np.where(y_data == 0): \n {np.where(y_data == 0)}")

print(f"np.where(y_data == 0)[0]: \n {np.where(y_data == 0)[0]}")

print(f"np.where(y_data == 0)[0][:3]: \n {np.where(y_data == 0)[0][:3]}")


print(f"class_1: \n {class_1}")
print(f"class_2: \n {class_2}")

# 取り出したインデックスを結合
selected_indices = np.concatenate([class_0, class_1, class_2])

print(f"selected_indices: \n {selected_indices}")

# 選択されたインデックスを使ってデータを抽出
X_data_selected = X_data[selected_indices]
y_data_selected = y_data[selected_indices]

print(f"X_data_selected: \n {X_data_selected}")
print(y_data_selected)
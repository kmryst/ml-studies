import numpy as np
from sklearn import datasets
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import math

def plot_decision_regions(x, y, model, x1_mesh, x2_mesh):
    markers = ('s', 'x', 'o') # matplotlib.markers 四角、バツ、マル
    cmap = ListedColormap(('red', 'blue', 'green')) # matplotlib.colors.ListedColormap

    ## メッシュデータ全部を学習モデルで分類
    z = model.predict(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T) # predictメソッドは各行が1サンプルの2次元配列を期待
    print(f"z.shape: {z.shape}")
    z = z.reshape(x1_mesh.shape)
    print(f"x1_mesh.shape: {x1_mesh.shape}")
    print(f"x2_mesh.shape: {x2_mesh.shape}")


    ## 決定境界を描いている
    plt.contourf(x1_mesh, x2_mesh, z, alpha=0.4, cmap=cmap) # contour: 輪郭 コンター
    plt.xlim(x1_mesh.min(), x1_mesh.max())
    plt.ylim(x2_mesh.min(), x2_mesh.max())

    """
    正解のmarkerをプロット
    ループ
    1回目: idx = 0, cl = 0
    2回目: idx = 1, cl = 1
    3回目: idx = 2, cl = 2
    """
    for idx, cl in enumerate(np.unique(y)): # indexとvalueを返す
        plt.scatter(x=x[y == cl, 0],
                    y=x[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx), # ListedColormapクラスは内部に__call__メソッドが定義されているため、インスタンスを関数のように呼び出すことができます
                    edgecolors='black',
                    marker=markers[idx],
                    label=cl)





data = datasets.load_iris()
x_data = data.data
y_data = data.target

class_0 = np.where(y_data == 0)[0][:3]
class_1 = np.where(y_data == 1)[0][:3]
class_2 = np.where(y_data == 2)[0][:3]

selected_indices = np.concatenate([class_0, class_1, class_2])

x_data = x_data[selected_indices]
y_data = y_data[selected_indices]

print(x_data)
print(y_data)


x_data = x_data[:, [0,1]]
# x_data = x_data[:, 0:2]
# x_data = x_data[:, :2]

print(x_data)



# 入力データの各変数が平均0,標準偏差1になるように正規化
# 各アルゴリズムのプロット結果を比較しやすいように予め全入力データを正規化

# インスタンス化
sc = StandardScaler()
# 訓練データとテストデータで同じ平均、標準偏差を使うためにここまで計算しておく
# それらの値をScalerオブジェクトに保存します
sc.fit(x_data)

# 保存された平均と標準偏差を使って、実際のデータ変換を行います
# 変換式：(x - 平均) / 標準偏差
x_data = sc.transform(x_data)

# 後でテストデータを変換する時も同じscalerを使用します
# x_test = sc.transform(x_test)

print(x_data)




x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                    y_data,
                                                    test_size=0.2)



x1_min, x1_max = x_data[:, 0].min()-1, x_data[:, 0].max()+1
x2_min, x2_max = x_data[:, 1].min()-1, x_data[:, 1].max()+1
x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                               np.arange(x2_min, x2_max, 0.1))

print(f"x1_mesh: \n {x1_mesh} \n ")
print(f"x2_mesh: \n {x2_mesh} \n ")



lr = LogisticRegression(C=10)
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='rbf', C=1.0)
dc = DecisionTreeClassifier(criterion='entropy', max_depth=3)
rf = RandomForestClassifier(criterion='entropy',
                            n_estimators=10)

models = [lr, knn, svm, dc, rf]
model_names = ['logistic regression',
               'k nearest neighbor',
               'svm',
               'decision tree',
               'random forest']



plt.figure(figsize=(8,6))
plot_num = 1
for model_name, model in zip(model_names, models):
    plt.subplot(math.ceil(len(models)/2), 2, plot_num)

    model.fit(x_train, y_train)
    
    plot_decision_regions(x_data, y_data, model, x1_mesh, x2_mesh)
    plt.title(model_name)
    plot_num += 1

plt.tight_layout()
plt.show()
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


#
# 決定境界プロット関数
#
def plot_decision_regions(x, y, model, resolution=0.01): # resolution: 解像度

    ## 今回は被説明変数が3クラスのため散布図のマーカータイプと3種類の色を用意
    ## クラスの種類数に応じて拡張していくのが良いでしょう
    markers = ('s', 'x', 'o') # matplotlib.markers
    cmap = ListedColormap(('red', 'blue', 'green')) # matplotlib.colors.ListedColormap

    ## 2変数の入力データの最小値から最大値まで引数resolutionの幅でメッシュを描く
    x1_min, x1_max = x[:, 0].min()-1, x[:, 0].max()+1
    x2_min, x2_max = x[:, 1].min()-1, x[:, 1].max()+1
    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                   np.arange(x2_min, x2_max, resolution))

    z = model.predict(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T) # Irisの予想(0, 1, 2)のいずれか
    
    print(z)
    
    z = z.reshape(x1_mesh.shape)
    
    print(f"z.reshape: {z}")

    ## メッシュデータと分離クラスを使って決定境界を描いている
    plt.contourf(x1_mesh, x2_mesh, z, alpha=0.4, cmap=cmap) # matplotlib.pyplot.contourf
    plt.xlim(x1_mesh.min(), x1_mesh.max())
    plt.ylim(x2_mesh.min(), x2_mesh.max())

    # for idx, cl in enumerate(np.unique(y)): # indexとvalueを返す
        # plt.scatter(x=x[y == cl, 0], # 散布図 y==clはブール配列、Trueのxのみ取り出す
        #             y=x[y == cl, 1],
        #             alpha=0.6,
        #             c=cmap(idx),
        #             edgecolors='black',
        #             marker=markers[idx],
        #             label=cl)

#
# データの取得
#
data = datasets.load_iris()
x_data = data.data
y_data = data.target


# 2変数だけを抽出
x_data = x_data[:, [0,1]]


sc = StandardScaler() # sklearn.preprocessing.StandardScaler
sc.fit(x_data) # 平均と標準偏差を計算
x_data = sc.transform(x_data) # 標準化(各特徴の平均が0、標準偏差が1になるようにデータを変換)


# データを学習用/テスト用に分割している
x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                    y_data,
                                                    test_size=0.2)

#
# 機械学習アルゴリズムの定義
#
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

#
# それぞれのモデルにおいて決定境界をプロット
#
plt.figure(figsize=(8,6))
plot_num = 1
for model_name, model in zip(model_names, models):

    plt.subplot(math.ceil(len(models)/2), 2, plot_num)
    # モデルの学習
    model.fit(x_train, y_train)
    # 決定境界をプロット
    plot_decision_regions(x_data, y_data, model)
    plt.title(model_name)
    plot_num += 1

plt.tight_layout()
plt.show()
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go

# データの準備
iris = datasets.load_iris()
X = iris.data[:, [0, 1]]  # 最初の2つの特徴量のみ使用
y = iris.target

# データの標準化
sc = StandardScaler()
X = sc.fit_transform(X)

# メッシュグリッドの作成
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x1_mesh, x2_mesh = np.meshgrid(
    np.arange(x1_min, x1_max, 0.1),
    np.arange(x2_min, x2_max, 0.1)
)

# モデルの学習と予測
model = LogisticRegression()
model.fit(X, y)

# メッシュグリッドポイントでの予測
z = model.predict(np.c_[x1_mesh.ravel(), x2_mesh.ravel()])
z = z.reshape(x1_mesh.shape)

# プロットの作成
fig = go.Figure()

# 決定境界の追加
fig.add_trace(go.Contour(
    x=np.arange(x1_min, x1_max, 0.1),
    y=np.arange(x2_min, x2_max, 0.1),
    z=z,
    colorscale='RdBu',
    opacity=0.5
))

# データポイントの追加
for cl in np.unique(y):
    mask = y == cl
    fig.add_trace(go.Scatter(
        x=X[mask, 0],
        y=X[mask, 1],
        mode='markers',
        name=f'Class {cl}',
        marker=dict(size=10)
    ))

# レイアウトの設定
fig.update_layout(
    title='Interactive Decision Boundary',
    xaxis_title='Feature 1',
    yaxis_title='Feature 2',
    showlegend=True
)

# プロットの表示
fig.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal # 多変量正規分布、多次元正規分布

N = 60
X = np.linspace(-3, 3, N) #-3から3を60分割
Y = np.linspace(-3, 4, N)
print(X)
X, Y = np.meshgrid(X, Y)
print(X)
print(X.shape)

# muはギリシャ文字のμ（ミュー）の英語表記で、統計学や確率論で平均を表すのによく使われます
mu = np.array([0., 1.]) # 多変量正規分布の中心がx軸上で0、y軸上で1
Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]]) # 共分散行列


# X座標の平面とY座標の平面が重なってる感じ
pos = np.empty(X.shape + (2,)) # (60, 60, 2)
pos[:, :, 0] = X
pos[:, :, 1] = Y

F = multivariate_normal(mu, Sigma)
Z = F.pdf(pos)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)

plt.show()
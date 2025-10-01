import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# 元のコードの設定
N = 60
X = np.linspace(-3, 3, N)
Y = np.linspace(-3, 4, N)
X, Y = np.meshgrid(X, Y)

mu = np.array([0., 1.])
Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])

pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

F = multivariate_normal(mu, Sigma)
Z = F.pdf(pos)

# 散布図用のデータ生成
num_points = 500
samples = F.rvs(num_points)

# プロット
fig = plt.figure(figsize=(16, 6))

# 3D表面プロット
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Probability Density')
ax1.set_title('Bivariate Gaussian Distribution - 3D Surface')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

# 散布図
ax2 = fig.add_subplot(122)
scatter = ax2.scatter(samples[:, 0], samples[:, 1], c='red', s=20, alpha=0.5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Bivariate Gaussian Distribution - Scatter Plot')

# 等高線の追加
contour = ax2.contour(X, Y, Z, levels=10, cmap='viridis')
fig.colorbar(contour, ax=ax2)

plt.tight_layout()
plt.show()
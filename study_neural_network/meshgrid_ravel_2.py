import numpy as np


x1 = np.arange(0, 6, 1)
x2 = np.arange(1, 4, 1)
print(f"x1: \n {x1}")
print(f"x2: \n {x2}")

x1_mesh, x2_mesh = np.meshgrid(x1, x2)
print(x1_mesh)
print(x2_mesh)

print(x1_mesh.ravel()) # ravel: ほぐす
print(x2_mesh.ravel())
print(np.array([x1_mesh.ravel(), x2_mesh.ravel()]))
print(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T)



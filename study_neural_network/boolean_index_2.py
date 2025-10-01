import numpy as np

x = np.arange(35)
print(f"np.arange(35): \n {x}\n")


x = np.arange(35).reshape(5, 7)
print(f".reshape(5, 7): \n {x}\n")


b = x > 20
print(f" x > 20: \n {b}\n")


print(f"b[:, 5]: \n {b[:, 5]}\n")


print(f"x[b[:, 5]]: \n {x[b[:, 5]]}\n")


print(f"x[b[:, 5]][:, -2:]: \n {x[b[:, 5]][:, -2:]}")

print(f"x[b[:, 5]][:, :2]: \n {x[b[:, 5]][:, :2]}")


import numpy as np

# シードを固定して同じ乱数を生成する例
np.random.seed(42)  # シード値を42に設定

# 1回目の乱数生成
x1 = np.random.rand(3)  # 3つの乱数を生成
print("1回目:", x1)    # 毎回同じ値が出力される

# 2回目の乱数生成 
x2 = np.random.rand(3)  # 別の3つの乱数を生成
print("2回目:", x2)    # 毎回同じ値だが1回目とは異なる

# シードを再設定すると最初から同じ順序で生成される
np.random.seed(42)      # 同じシード値を設定
x3 = np.random.rand(3)  # 1回目と同じ値が生成される
print("再設定後:", x3)  # x1と同じ値が出力される
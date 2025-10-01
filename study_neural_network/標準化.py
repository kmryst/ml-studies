from sklearn import datasets
from sklearn.preprocessing import StandardScaler

data = datasets.load_iris()
x_data = data.data

x_data = x_data[:5] 

print(x_data)

sc = StandardScaler()
sc.fit(x_data) 
x_data = sc.transform(x_data) 

print(x_data)

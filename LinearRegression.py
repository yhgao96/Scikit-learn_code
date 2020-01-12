from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

loaded_data=datasets.load_boston()  #波士顿房价样本
data_X=loaded_data.data
data_y=loaded_data.target

model=LinearRegression()      #线性回归
model.fit(data_X,data_y)

print(model.predict(data_X[:4,:]))
print(data_y[:4])

#产生样本数据
X,y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)
plt.scatter(X,y)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.ensemble import AdaBoostRegressor

data = load_boston()
train_x, test_x, train_y,test_y = train_test_split(data.data, data.target, test_size=0.25, random_state=33)

regressor = AdaBoostRegressor()
regressor.fit(train_x, train_y)
predict = regressor.predict(test_x)
mse = mean_squared_error(test_y,predict)

print("房价预测结果 ", predict)
print("均方差 ", round(mse, 2))

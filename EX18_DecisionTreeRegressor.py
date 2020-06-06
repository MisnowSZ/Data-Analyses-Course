from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
import graphviz

boston = load_boston()
#print(boston)
features = boston.data
prices = boston.target
#print(prices)

train_features, test_features, train_price, test_price = train_test_split(features, prices, test_size=0.33)

dtr = DecisionTreeRegressor()

dtr.fit(train_features, train_price)

predict_price = dtr.predict(test_features)
print("predict price = ", predict_price)
print("回归树二乘偏差均值:", mean_squared_error(test_price, predict_price))
print("回归树绝对偏差均值:", mean_absolute_error(test_price, predict_price))

dot_data = export_graphviz(dtr, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("Boston")

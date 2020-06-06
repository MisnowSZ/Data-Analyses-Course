from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

iris=load_iris()

print(iris)

features = iris.data
labels = iris.target

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=0)
print(train_features, test_features, train_labels, test_labels)

clf = DecisionTreeClassifier(criterion = 'gini')
clf = clf.fit(train_features, train_labels)

print(clf)

test_predict = clf.predict(test_features)

print(test_predict)

score = accuracy_score(test_labels, test_predict)

print("score = %f" % score)

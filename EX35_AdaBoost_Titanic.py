from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd 
from sklearn.feature_extraction import DictVectorizer

train_data = pd.read_csv('C:\\Users\\qye\\Python\\Learning\\Titanic_Data-master\\train.csv')
test_data = pd.read_csv('C:\\Users\\qye\\Python\\Learning\\Titanic_Data-master\\test.csv')

train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
#print(train_data['Embarked'].value_counts())

train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)

#print(train_data.info())
#print('--'*30)

#select feature
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

dvec = DictVectorizer(sparse=False)
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
test_features = dvec.fit_transform(test_features.to_dict(orient='record'))

clf = AdaBoostClassifier()
clf.fit(train_features, train_labels)
predict = clf.predict(test_features)
#mse = mean_squared_error(test_y,predict)

print("Survival Prediction:  ", predict)
#print("均方差 ", round(mse, 2))

acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print("acc sore %.4lf" %acc_decision_tree)

import pandas as pd
import graphviz

train_data = pd.read_csv('C:\\Users\\qye\\Python\\Learning\\Titanic_Data-master\\train.csv')
test_data = pd.read_csv('C:\\Users\\qye\\Python\\Learning\\Titanic_Data-master\\test.csv')

#print(train_data.info())
#print('--'*30)

#print(train_data.describe())
#print('++'*30)

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
print(test_data.info())
print('--'*30)

from sklearn.feature_extraction import DictVectorizer
dvec = DictVectorizer(sparse=False)
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
test_features = dvec.fit_transform(test_features.to_dict(orient='record'))
print (dvec.feature_names_)

#build ID3 decision tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(train_features, train_labels)

pred_labels = clf.predict(test_features)

acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print("acc sore %.4lf" %acc_decision_tree)

import numpy as np
from sklearn.model_selection import cross_val_score
print("cross_val_score = %4lf" % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))

from sklearn import tree

dot_data = tree.export_graphviz(clf, out_file = None)
graph = graphviz.Source(dot_data)
graph.render("tree")
graph.view('graph')

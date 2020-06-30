from sklearn import svm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("C:\\Users\\qye\\Python\\Learning\\breast_cancer_data-master\\data.csv")
pd.set_option('display.max_columns', None)
#print(data.columns)
#print(data.head(5))
#print(data.describe())

feature_mean = list(data.columns[2:12])
feature_se = list(data.columns[12:22])
feature_worst = list(data.columns[22:32])

data.drop("id", axis=1, inplace=True)
data['diagnosis']=data['diagnosis'].map({'M':1, 'B':0})

sns.countplot(data['diagnosis'], label='Count')
plt.show()

#用热力图呈现feature_mean字段之间的相关性
corr = data[feature_mean].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr, annot=True)
plt.show()


# 特征选择
features_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean'] 

#抽取30%作为测试集
train, test = train_test_split(data, test_size = 0.3)

train_x = train[features_remain]
train_y = train['diagnosis']
test_x = test[features_remain]
test_y = test['diagnosis']

ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)

#创建SVM分类器
model = svm.SVC()
model.fit(train_x, train_y)

prediction = model.predict(test_x)
print("Accuracy: ", metrics.accuracy_score(test_y, prediction))

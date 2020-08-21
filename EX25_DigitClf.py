from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

digits = load_digits()
data = digits.data
#print(digits)
#print(digits.images[0])
#print(digits.target[0])
#plt.gray()
#plt.imshow(digits.images[0])
#plt.show()

train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size = 0.25, random_state = 33)
#Z-score 规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.fit_transform(test_x)

#KNN Classifier
knn = KNeighborsClassifier()
knn.fit(train_ss_x, train_y)
predict_y = knn.predict(test_ss_x)
print("Knn accuracy: %.4f " % accuracy_score(test_y, predict_y))

#SVM Classifier
svm_model = svm.SVC()
svm_model.fit(train_ss_x, train_y)
svm_predict = svm_model.predict(test_ss_x)
print("SVM accuracy: %.4f " % accuracy_score(test_y, svm_predict))

#Beyes
#不能有负数
mm = preprocessing.MinMaxScaler()
train_mm_x = mm.fit_transform(train_x)
test_mm_x = mm.fit_transform(test_x)
clf = MultinomialNB(alpha=0.0001).fit(train_mm_x, train_y)
predict_vec = clf.predict(test_mm_x)
print("Bayes accuracy: %.4f " % accuracy_score(test_y, predict_vec))

#CART 决策树
clf_cart = DecisionTreeClassifier(criterion='gini')
clf_cart = clf_cart.fit(train_ss_x, train_y)
predict_cart = clf_cart.predict(test_ss_x)
print("CART accuracy: %.4f " % accuracy_score(test_y, predict_cart))

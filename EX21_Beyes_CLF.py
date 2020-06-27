from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import os
import jieba

stop_words_path = 'C:\\Users\\qye\\Python\\Learning\\text_classification-master\\text classification\\stop\\stopword.txt'

def split_words(str):
    """
    Split the text to words with space
    str: input to be splited string
    return: splited string with space
    """
    new_str = ""
    words = jieba.cut(str)
    for word in words:
        new_str += word + ' '
    return new_str
    
def load_file(dir, label):
    """
    load files in dir
    dir: the dir of the text files
    label: the label
    return: text content
    """
    file_list = os.listdir(dir)
    word_list = []
    label_list = []
    for file in file_list:
        file_path = dir + '/' + file
        text = open(file_path, 'r', encoding='gb18030').read()
        word_list.append(split_words(text))
        label_list.append(label)

    return word_list, label_list

# 1 文档分词
"""
format example:
documents = [
    'this is the bayes document',
    'this is the second second document',
]
"""
train_words_female, train_labels_female = load_file('C:\\Users\\qye\\Python\\Learning\\text_classification-master\\text classification\\train\\女性', '女性')
#print(train_words_female)
train_words_physics, train_labels_physics = load_file('C:\\Users\\qye\\Python\\Learning\\text_classification-master\\text classification\\train\\体育', '体育')
train_words_literature, train_labels_literature = load_file('C:\\Users\\qye\\Python\\Learning\\text_classification-master\\text classification\\train\\文学', '文学')
train_words_university, train_labels_university = load_file('C:\\Users\\qye\\Python\\Learning\\text_classification-master\\text classification\\train\\校园', '校园')

train_words_list = train_words_female + train_words_physics + train_words_literature + train_words_university
train_labels_list = train_labels_female + train_labels_physics + train_labels_literature + train_labels_university

# load test data
test_words_female, test_labels_female = load_file('C:\\Users\\qye\\Python\\Learning\\text_classification-master\\text classification\\test\\女性', '女性')
test_words_physics, test_labels_physics = load_file('C:\\Users\\qye\\Python\\Learning\\text_classification-master\\text classification\\test\\体育', '体育')
test_words_literature, test_labels_literature = load_file('C:\\Users\\qye\\Python\\Learning\\text_classification-master\\text classification\\test\\文学', '文学')
test_words_university, test_labels_university = load_file('C:\\Users\\qye\\Python\\Learning\\text_classification-master\\text classification\\test\\校园', '校园')

test_words_list = test_words_female + test_words_physics + test_words_literature + test_words_university
test_labels_list = test_labels_female + test_labels_physics + test_labels_literature + test_labels_university

# 2 加载停用词
stop_words = open(stop_words_path, 'r', encoding='utf-8').read() #return a string
stop_words = stop_words.encode('utf-8').decode('utf-8-sig') #列表头部\ufeff处理？？？？？？
stop_words = stop_words.split('\n')
#print(stop_words)

# 3 计算单词的权重
tfidf_vec = TfidfVectorizer(stop_words = stop_words, max_df = 0.5)
tfidf_feature_train = tfidf_vec.fit_transform(train_words_list)
#print("不重复的词: ", tfidf_vec.get_feature_names())
#print("每个词的ID: ", tfidf_vec.vocabulary_)
#print(tfidf_matrix_female)

tfidf_feature_test = tfidf_vec.transform(test_words_list) #前面fit过，这里只需要transform

# 4 生成朴素贝叶斯分类器
clf = MultinomialNB(alpha = 0.001).fit(tfidf_feature_train, train_labels_list)

# 5 使用分类器进行预测
predict_labels = clf.predict(tfidf_feature_test)
print(predict_labels)

# 6 计算准确率
print('Accuracy socre: ', metrics.accuracy_score(test_labels_list, predict_labels))

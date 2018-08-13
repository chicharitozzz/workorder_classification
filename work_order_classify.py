# @Time : 2018/8/2 11:33 
# @Author : Chicharito_Ron
# @File : work_order_classify.py
# @Software: PyCharm Community Edition
# 短文本工单分类
# 词袋模型
import json
import numpy as np
import jieba
import re
import string
import multiprocessing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.externals import joblib

def clean_data(fname):
    """读取数据集，并进行数据清洗"""
    with open(fname, encoding='utf-8') as f:
        data = json.load(f)

    data = np.array(data)
    X = data[:, 0]
    Y = data[:, 1]

    # 删去无标签数据
    none_labels = []
    for i in range(Y.shape[0]):
        if not Y[i]:
            #         print(X[i])
            #         print('*'*80)
            none_labels.append(i)

    Y = np.delete(Y, none_labels)
    X = np.delete(X, none_labels)

    # 删去#号开头的文本和空文本
    invaild_x = []
    for i in range(X.shape[0]):
        if not X[i] or X[i].startswith('#'):
            invaild_x.append(i)

    Y = np.delete(Y, invaild_x)
    X = np.delete(X, invaild_x)

    # 删去废单和回访工单
    discard_x = []
    for i in range(X.shape[0]):
        if X[i].strip().startswith('废单') or X[i].strip().startswith('回访工单'):
            discard_x.append(i)

    Y = np.delete(Y, discard_x)
    X = np.delete(X, discard_x)

    # 删去非受理范围工单
    non_acceptable = np.nonzero(Y[:] == '非受理范围')
    Y = np.delete(Y, non_acceptable)
    X = np.delete(X, non_acceptable)

    # 有的无效电话不属于非受理范围
    invaild_p = []
    for i in range(X.shape[0]):
        if X[i].startswith('无效电话'):
            invaild_p.append(i)

    Y = np.delete(Y, invaild_p)
    X = np.delete(X, invaild_p)

    return X, Y


def get_stop_words(fname):
    """加载停用词列表"""
    stop_words = []
    with open(r'./static/停用词.txt', encoding='utf-8-sig') as f:
        lines = f.readlines()

    for line in lines:
        stop_words.append(line.strip())

    return stop_words


def judge(word):
    """判断词是否由英文字母加数字组成"""
    for w in word:
        if w not in string.printable:
            return True

    return False


def pro_process(fname, X, Y, s_words):
    """预处理：分词，去停用词"""
    jieba.load_userdict(fname)  # 加载分词词典

    data_list = []
    revisit_x = []  # 删除回访工单

    for i in range(X.shape[0]):
        x = X[i]
        # 处理特殊工单:催单、补单、撤单、撤销
        if x.strip().startswith(('催单', '补单', '撤单', '撤销', '催办', '转接')):
            quert_l = re.split(r'\n+', x.strip())  # 以回车符分割字符串

            if len(quert_l) == 3:  # 催单、问题描述：、来电目的：
                q = re.findall('：(.*)', quert_l[1])[0].strip()  # 匹配问题描述后文本

            elif len(quert_l) == 1:  # 催单：
                q = re.findall('：(.*)', quert_l[0])[0].strip()  # 匹配冒号后文本

            else:  # 催单:\n
                q = quert_l[1].strip()

        else:  # 处理正常工单
            quert_l = re.split(r'\n+', x.strip())  # 以回车符分割字符串
            match_l = re.findall('：(.*)', quert_l[0])  # 匹配冒号后文本;1.问题描述：
            if not match_l:
                q = quert_l[0].strip()
            else:
                q = match_l[0].strip()

        # 分词
        seg_sentence = jieba.cut(q, cut_all=False)
        tmp_l = []

        # 遍历停用词表,去停用词
        for w in seg_sentence:
            if w not in s_words and len(w) >= 2 and not w.isdigit() and judge(w):  # TODO:判断词是否为小数
                tmp_l.append(w)

        # TODO:在数据清洗时删除
        if not tmp_l:  # 便民服务回访、工单回访只有工单号，无实际内容，删除
            revisit_x.append(i)
            # print(x)
            # print('切分后',q)
            # print('*'*50)

        data_list.append(' '.join(tmp_l))

    data_list = np.array(data_list)
    data_list = np.delete(data_list, revisit_x)
    Y = np.delete(Y, revisit_x)

    return data_list, Y


class Textclassify:
    def __init__(self, X, Y, stop_words):
        self.X = X
        self.Y = Y
        self.s_words = stop_words

    def cross_validation(self, X_vec):
        """交叉验证，90%数据用于训练，10%数据用于测试"""
        data = list(zip(X_vec, self.Y))
        sep = int(len(data) * 0.9)
        np.random.shuffle(data)  # 数据乱序

        train_data = data[:sep]
        test_data = data[sep:]

        X_train, Y_train = zip(*train_data)
        X_test, Y_test = zip(*test_data)

        return X_train, Y_train, X_test, Y_test

    def proprocessing(self):
        """数据预处理：向量化；计算TF-IDF权重"""
        vectorizer = CountVectorizer(stop_words=self.s_words, min_df=1e-3)
        vec_train = vectorizer.fit_transform(self.X)
        print('特征数量为:{}'.format(len(vectorizer.get_feature_names())))

        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vec_train)
        print('文本向量维度:{}'.format(tfidf.shape))
        tfidf = tfidf.toarray()
        return tfidf

    @staticmethod
    def lower_dimension(tfidf):
        """PCA降维"""
        pca = PCA(n_components=100, copy=False)  # 向量维度
        X_lowd = pca.fit_transform(tfidf)

        return X_lowd

    def classify(self):
        """分类"""
        acc_li = []
        clf = RandomForestClassifier(n_estimators=100)  # 参数设置
        tfidf = self.proprocessing()
        X_lowd = self.lower_dimension(tfidf)

        for i in range(10):
            X_train, Y_train, X_test, Y_test = self.cross_validation(X_lowd)
            clf.fit(X_train, Y_train)
            acc = clf.score(X_test, Y_test)
            print(acc)
            acc_li.append(acc)

        return acc_li

    def get_prob(self):
        """获取分类概率"""
        tfidf = self.proprocessing()
        X_lowd = self.lower_dimension(tfidf)
        acc_li = []

        clf = RandomForestClassifier(n_estimators=100, n_jobs=multiprocessing.cpu_count() - 2,
                                     class_weight='balanced')  # 参数设置

        X_train, Y_train, X_test, Y_test = self.cross_validation(X_lowd)
        clf.fit(X_train, Y_train)
        # print(clf.classes_)
        p = clf.predict_proba(X_test[:5])
        return p

    def class_report(self):
        """获取完整的分类报告:精确率、召回率等"""
        tfidf = self.proprocessing()
        X_lowd = self.lower_dimension(tfidf)
        acc_li = []

        clf = RandomForestClassifier(n_estimators=120, n_jobs=multiprocessing.cpu_count() - 2,
                                     class_weight='balanced')  # 参数设置
        X_train, Y_train, X_test, Y_test = self.cross_validation(X_lowd)
        clf.fit(X_train, Y_train)

        #         joblib.dump(clf, 'rfmodel.pkl')  # 存储模型

        Y_pred = clf.predict(X_test)
        report = classification_report(Y_test, Y_pred)
        print(report)
        return report


if __name__ == '__main__':
    X, Y = clean_data('../工单数据/2018年数据.json')
    print(len(X))
    print(len(Y))
    # s_words = get_stop_words('./static/停用词.txt')
    # data_list, Y = pro_process('./static/my_dict.txt', X, Y)
    # tc = Textclassify(data_list, Y, s_words)
    # acc = tc.classify()

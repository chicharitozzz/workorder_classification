# @Time : 2018/8/12 15:26 
# @Author : Chicharito_Ron
# @File : order_process.py
# @Software: PyCharm Community Edition
# 向量化单个工单
# TODO:使用TfidfVectorizer
import jieba
import string
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA


# 更新停用词列表
def get_stop_words():
    """停用词列表"""
    stop_words = []
    with open('./static/停用词.txt', encoding='utf-8-sig') as f:
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


def processing(order):
    s_words = get_stop_words()
    # 分词
    jieba.load_userdict('./static/my_dict.txt')
    seg_sentence = jieba.cut(order, cut_all=False)
    # 去停用词
    text_l = []
    for w in seg_sentence:
        if w not in s_words and len(w) >= 2 and not w.isdigit() and judge(w):
            text_l.append(w)

    text = ' '.join(text_l)

    # 计算TF-IDF权重,降维
    with open('./static/corpus.json', encoding='utf-8') as f:
        data_list = json.load(f)

    data_list.append(text)

    vectorizer = CountVectorizer(min_df=1e-3)
    vec_train = vectorizer.fit_transform(data_list)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vec_train)
    tfidf = tfidf.toarray()
    pca = PCA(n_components=100, copy=False)  # 向量维度
    X_lowd = pca.fit_transform(tfidf)

    return X_lowd[-1]


if __name__ == '__main__':
    r = processing('市民咨询二手房公积金贷款对房屋面积是否有限制。')
    print(r)

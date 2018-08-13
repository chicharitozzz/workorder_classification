# @Time : 2018/8/12 14:46 
# @Author : Chicharito_Ron
# @File : main.py.py 
# @Software: PyCharm Community Edition

from sklearn.externals import joblib
from flask import Flask, request, render_template
import order_process

app = Flask(__name__)


def rf(model_pos, order):
    order_vec = order_process.processing(order)  # 向量化工单

    clf = joblib.load(model_pos)
    classes = clf.classes_

    label = clf.predict(order_vec.reshape(1, -1))
    prob = clf.predict_proba(order_vec.reshape(1, -1))

    return label.tolist(), prob.tolist(), classes.tolist()


@app.route('/', methods=['GET'])
def pred_label(model_pos):
    if request.method == 'GET':
        if request.args.get('order'):
            order = request.args.get('order')
            label, prob, classes = rf(model_pos, order)
            return render_template('predict.html', res=label[0], prob=prob[0], cls=classes)
        return render_template('predict.html')


if __name__ == '__main__':
    app.run(port=8082, debug=True)
    # l, p, c = rf('../工单数据相关/rfmodel.pkl', '市民咨询二手房公积金贷款对房屋面积是否有限制。')
    # print(l)
    # print(p)
    # print(c)

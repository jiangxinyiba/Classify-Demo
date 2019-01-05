# 利用Bagging策略去改进决策树，构造集成树
import numpy as np
# import scipy as sp
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn import datasets
import matplotlib.pyplot as plt
import pylab as pl
import pydotplus
# 构造待分类数据集
def createData(sizeoftest):
    Data = [[1, 0, 1, 0, 0, 0, 'yes'],
            [1, 0, 0, 0, 0, 0, 'yes'],
            [0, 2, 2, 0, 2, 1, ' no'],
            [2, 2, 2, 2, 2, 0, ' no'],
            [0, 0, 1, 0, 0, 0, 'yes'],
            [2, 0, 0, 0, 0, 0, 'yes'],
            [0, 1, 0, 1, 0, 0, ' no'],
            [1, 1, 1, 1, 1, 0, ' no'],
            [2, 0, 0, 2, 2, 1, ' no'],
            [0, 0, 0, 0, 0, 0, 'yes'],
            [2, 1, 1, 1, 0, 0, ' no'],
            [0, 1, 0, 0, 1, 1, 'yes'],
            [1, 1, 0, 1, 1, 1, 'yes'],
            [1, 1, 0, 0, 1, 0, 'yes'],
            [1, 1, 1, 1, 1, 1, 'yes'],
            [1, 1, 0, 0, 1, 1, ' no'],
            [2, 0, 0, 2, 2, 0, ' no'],
            [0, 0, 1, 1, 1, 0, ' no']]
    Data = np.array(Data)
    X = Data[:,:-1]
    Y = Data[:,-1]
    for i in range(Y.shape[0]):
        if Y[i] == "yes" :
            Y[i] = 1
        else:
            Y[i] = 0.5

    #''''' 拆分训练数据与测试数据 '''
    x_train, x_test, y_train, y_test = train_test_split(X, Y,test_size=sizeoftest, random_state=0)
    return x_train, y_train, x_test, y_test

def DecisionTree(x_train,y_train,x_test, y_test):
    # ''''' 使用信息熵作为划分标准，对决策树进行训练 '''
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(x_train, y_train)
    Yp = clf.predict(x_test)

    # ''''' 把决策树结构写入文件 '''
    with open("tree.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("西瓜数据3.0的决策树.pdf")

    return Yp

    # ''''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
    # print(clf.feature_importances_)

# 样本重采样
def ReSampleforBagging(nSize,percent):
    # percent：每次抽取的样本百分比
    index = np.random.randint(nSize,size=int(nSize*percent))
    index = index.tolist()
    return index

# 分析结果
def result(y_test,Yp):
    width = 0.35
    count = 0
    Nte = Yp.shape[0]
    for i in range(Nte):
        if Yp[i] == y_test[i]:
            count += 1
    print("预测准确率：%s"   %(count/Nte))
    # plt.figure()
    # ind = np.arange(Nte)
    # Yp = list(map(eval, Yp))               # 将list的str转int ！！！
    # y_test = list(map(eval, y_test))
    # plt.bar(ind-width / 2, y_test, width, color='SkyBlue', label=u"真实值")
    # plt.bar(ind + width / 2, Yp, width, color='IndianRed', label=u"预测值")
    # plt.legend()
    # pl.rcParams['font.sans-serif'] = ['SimHei']
    # plt.show()

def count_list(X):
    unique_dict = {}
    for e in X:
        if e in unique_dict:
            unique_dict[e] += 1
        else:
            unique_dict[e] = 1
    return unique_dict

# main
if __name__ == '__main__':
    # 生成数据集
    # iris = datasets.load_iris()
    x_train, y_train, x_test, y_test = createData(0.4)

    print("真实值：")
    print(y_test)
    # 决策树法
    Yp = []
    Yp = DecisionTree(x_train,y_train,x_test, y_test)
    print("决策树法：")
    print(Yp)
    result(y_test, Yp)

    # Bagging改进决策树法
    Yp_sub = []
    Yp_all = []
    Yp_bagging = []
    nSize = x_train.shape[0]
    nTestSize = y_test.shape[0]
    M = 20            # 子模型个数
    Percent = 0.632    # 每个子模型重复采样的百分比
    for i in range(M):
        Index = ReSampleforBagging(nSize, Percent)
        Yp_sub = DecisionTree(x_train[Index], y_train[Index], x_test, y_test)
        # Yp_all = np.append(Yp_all,Yp_sub,axis = 0)
        Yp_all.append(Yp_sub.tolist())
    Yp_all = np.array(Yp_all)
    # 投票法确定集成输出
    for i in range(nTestSize):
        dict = count_list(Yp_all[:,i])
        Yp_bagging.append(max(dict, key=dict.get))
    Yp_bagging = np.array(Yp_bagging)
    print("Bagging改进决策树法：")
    print(Yp_bagging)
    result(y_test, Yp_bagging)







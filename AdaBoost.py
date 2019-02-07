# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 09:29:18 2017
Adaboost
@author: liujiping
"""
import numpy as np
# import Bagging_DecisionTree as BDT
from sklearn import datasets
import matplotlib.pyplot as plt

def loadSimData():
    '''
    输入：无
    功能：提供一个两个特征的数据集
    输出：带有标签的数据集
    '''
    datMat = np.matrix([[1. ,2.1],[2. , 1.1],[1.3 ,1.],[1. ,1.],[2. ,1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def stumpClassify(dataMatrix,dimen,thresholdValue,thresholdIneq):
    '''
    输入：数据矩阵，特征维数，某一特征的分类阈值，分类不等号
    功能：输出决策树桩标签[单层等间隔划分决策树]
    输出：标签
    '''
    returnArray =   (np.ones((np.shape(dataMatrix)[0],1)))
    if thresholdIneq == 'lt':
        returnArray[dataMatrix[:,dimen] <= thresholdValue] = -1
    else:
        lab = dataMatrix[:,dimen] > thresholdValue
        returnArray[lab] = -1
    return returnArray

def buildStump(dataArray,classLabels,D):
    '''
    输入：数据矩阵，对应的真实类别标签，特征的权值分布
    功能：在数据集上，找到加权错误率（分类错误率）最小的单层决策树，显然，该指标函数与权重向量有密切关系
    输出：最佳树桩（特征，分类特征阈值，不等号方向），最小加权错误率，该权值向量D下的分类标签估计值
    '''
    dataMatrix = np.mat(dataArray); labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    stepNum = 10.0; bestStump = {}; bestClassEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin)/stepNum
        for j in range(-1, int(stepNum)+1):
            for thresholdIneq in ['lt', 'gt']:
                thresholdValue =  rangeMin + float(j) * stepSize
                predictClass = stumpClassify(dataMatrix,i,thresholdValue,thresholdIneq)
                errArray =  np.mat(np.ones((m,1)))
                errArray[predictClass == labelMat] = 0
                weightError = D.T * errArray
                # print("split: dim %d, thresh: %.2f,threIneq:%s,weghtError %.3F" %(i,thresholdValue,thresholdIneq,weightError))
                if weightError < minError:
                    minError = weightError
                    bestClassEst = predictClass.copy()
                    bestStump['dimen'] = i
                    bestStump['thresholdValue'] = thresholdValue
                    bestStump['thresholdIneq'] = thresholdIneq
    return bestClassEst, minError, bestStump

def adaBoostTrainDS(dataArray,classLabels,numIt=40):
    '''
    输入：数据集，标签向量，最大迭代次数
    功能：创建adaboost加法模型
    输出：多个弱分类器的数组
    '''
    weakClass = []#定义弱分类数组，保存每个基本分类器bestStump
    m,n = np.shape(dataArray)
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        print("i: %d" %(i))
        # step1:找到最佳的单层决策树
        bestClassEst, minError, bestStump = buildStump(dataArray,classLabels,D)
        print("D.T: %s" %(D.T))
        # step2: 更新alpha
        alpha = float(0.5*np.log((1-minError)/max(minError,1e-16)))
        print("alpha: %s" %(alpha))
        bestStump['alpha'] = alpha
        # step3:将基本分类器添加到弱分类的数组中
        weakClass.append(bestStump)
        print("classEst: %s" %bestClassEst)
        # step4:更新权重，该式是让D服从概率分布
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,bestClassEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()             # 归一化
        #steo5:更新累计类别估计值
        aggClassEst += alpha*bestClassEst
        print ("aggClassEst: %s" %aggClassEst.T)
        print(np.sign(aggClassEst) != np.mat(classLabels).T)
        aggError = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        print("aggError: %s" %aggError)
        aggErrorRate = aggError.sum()/m
        print("total train error: %.3f" %aggErrorRate)
        if aggErrorRate == 0.0: break
    return weakClass,aggClassEst

def adaTestClassify(dataToClassify,labelToClassify,weakClass):
    dataMatrix = np.mat(dataToClassify)
    labelMattix = np.mat(labelToClassify)
    m =np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(weakClass)):
        classEst = stumpClassify(dataMatrix,weakClass[i]['dimen'],weakClass[i]['thresholdValue']\
                                 ,weakClass[i]['thresholdIneq'])
        aggClassEst += weakClass[i]['alpha'] * classEst
        prediction = np.sign(aggClassEst)
        # print(aggClassEst)
    # 计算预测结果
    error = np.mat(np.ones((m,1)))
    print("total test error: %.3f" %(error[prediction!=labelMattix.T].sum()/m))
    return aggClassEst

def plotROC(predStrengths, classLabels):
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(np.array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = np.argsort(predStrengths,axis=0)#get sorted index, it's reverse
    # sortindex = []
    # for index in sortedIndicies.tolist():
    #     index = index[0]
    #     sortindex.append(index)
    # sortindex = np.array(sortindex)
    # classLabels = np.array(classLabels)
    # classLabels = classLabels[sortindex]
    # predStrengths = predStrengths[sortedIndicies]
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist():
        index = index[0]
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)

if __name__  ==  '__main__':
    # 仿真数据分类
    # D =np.mat(np.ones((5,1))/5)
    # dataMatrix ,classLabels= loadSimData()
    # bestClassEst, minError, bestStump = buildStump(dataMatrix,classLabels,D)
    # print(bestStump)
    # weakClass = adaBoostTrainDS(dataMatrix,classLabels,9)
    # testClass = adaTestClassify(np.mat([0,0]),-1,weakClass)

    # 马疝数据集分类
    dataArr_tr,labelArr_tr = loadDataSet("horseColicTraining2.txt")
    classifier,aggClassEst_tr = adaBoostTrainDS(dataArr_tr, labelArr_tr, 50)
    dataArr_te, labelArr_te = loadDataSet("horseColicTest2.txt")
    aggClassEst_te = adaTestClassify(dataArr_te,labelArr_te, classifier)

    # 测试数据ROC曲线
    plotROC(aggClassEst_te, labelArr_te)
    # 训练数据ROC曲线
    plotROC(aggClassEst_tr, labelArr_tr)
    # scikit方法绘制
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, threshold = roc_curve(np.array(labelArr_te),np.array(aggClassEst_te))  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)
    print("the Area Under the Curve of Scikit is: ",roc_auc)
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
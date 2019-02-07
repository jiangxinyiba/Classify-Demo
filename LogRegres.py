'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *
import numpy as np
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+math.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))

    h = []
    for k in range(maxCycles):              #heavy on matrix operations
        Tdata_w = dataMatrix * weights      #更新权值，从而改变梯度
        h = [sigmoid(i) for i in Tdata_w]
        h = np.mat(h).T
        #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.T* error #matrix mult
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = np.array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    y = np.array(y).transpose()
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * np.array(dataMatrix[i])
    return weights

def stocGradAscent1_gai(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    W0 = [1]
    W1 = [1]
    W2 = [1]
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * np.array(dataMatrix[randIndex])
            W0.append(weights[0])
            W1.append(weights[1])
            W2.append(weights[2])
            del dataIndex[randIndex]
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(3,1,1)
    plt.plot(W0)
    plt.ylabel('W0')
    ax2 = plt.subplot(3, 1, 2)
    plt.plot(W1)
    plt.ylabel('W1')
    ax3 = plt.subplot(3, 1, 3)
    plt.plot(W2)
    plt.ylabel('W2')
    return weights

#
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            mw = sum(dataMatrix[randIndex] * weights)
            h = sigmoid(mw)
            # print("i=%d" %i)
            # print("j=%d" %j)
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * np.array(dataMatrix[randIndex])
            del dataIndex[randIndex]
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    from sklearn import preprocessing
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    # 构造训练数据
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 训练输入归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    trainingSet_norm = min_max_scaler.fit_transform(np.array(trainingSet))
    # 训练逻辑回归的权值
    trainWeights = stocGradAscent1(trainingSet_norm, trainingLabels, 100)
    errorCount = 0; numTestVec = 0.0
    # 构造测试数据
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 测试输入归一化
        lineArr_norm = min_max_scaler.transform(np.array(lineArr))
        # 对测试数据进行分类并评判记录结果
        if int(classifyVector(np.array(lineArr_norm), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

##################################################################################
if __name__ == '__main__':
    #梯度上升法
    dataMat,  labelMat = loadDataSet()
    Wga = gradAscent(dataMat, labelMat)
    print("梯度上升法求得的权值：[%.3f %.3f %.3f]" %(Wga[0],Wga[1],Wga[2]))
    #plotBestFit(Wga)
    #随机梯度上升法
    Wsga = stocGradAscent0(dataMat, labelMat)
    print("随机梯度上升法求得的权值：[%.3f %.3f %.3f]" %(Wsga[0],Wsga[1],Wsga[2]))
    #plotBestFit(Wsga)
    # 改进随机梯度上升法
    Wsga1 = stocGradAscent1_gai(dataMat, labelMat, numIter=150)
    print("改进随机梯度上升法求得的权值：[%.3f %.3f %.3f]" % (Wsga1[0], Wsga1[1], Wsga1[2]))
    #plotBestFit(Wsga1)

    # 马疝气病分类问题
    multiTest()
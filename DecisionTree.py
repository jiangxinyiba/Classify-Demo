import numpy as np
import math

# 定义按照某个特征进行划分的函数 splitDataSet
# 输入三个变量（待划分数据集， 第axis个特征，是否为value值)
# PS：因为计算熵的时候只需要考虑各类样本个数，不考虑属性个数
#    所以这个函数可以在计算熵的时候用，也需要在后续确定分支子集的时候用
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 剔除第axis列特征
            reduceFeatVec = featVec[:axis]              # 选择第axis列之前的特征
            reduceFeatVec.extend(featVec[axis + 1:])    # 选择第axis列之后的特征
            retDataSet.append(reduceFeatVec)
    return retDataSet                                   # 返回不含划分特征的子集

# 求变量的信息熵,把属性的分类标识传入即可
def calEntropy(dataSet):
    nEntries = len(dataSet)
    # 为分类创建字典,确定每个类别的样本个数classCounts
    classCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in classCounts.keys():
            classCounts.setdefault(currentLabel, 0)
        classCounts[currentLabel] += 1
    # 计算香农墒
    shannonEnt = 0.0
    for key in classCounts:
        prob = float(classCounts[key]) / nEntries
        shannonEnt += -1 * prob * math.log2(prob)
    return shannonEnt

#  定义按照最大信息增益划分数据的函数
def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1  # 属性个数
    print(numFeature)
    baseEntropy = calEntropy(dataSet) # 根节点信息熵
    bestInforGain = 0
    bestFeature = -1

    for i in range(numFeature):
        featList = [number[i] for number in dataSet] #得到某个特征下所有值
        uniqualVals = set(featList) #set无重复的属性特征值
        newEntrogy = 0
        #求和
        for value in uniqualVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet)) #即p(t)
            newEntrogy += prob * calEntropy(subDataSet) #对各子集求香农墒

        infoGain = baseEntropy - newEntrogy #计算信息增益
        print(infoGain)

        # 比较信息增益，保留最大的信息增益对应的属性特征
        if infoGain > bestInforGain:
            bestInforGain = infoGain
            bestFeature = i
    return bestFeature

# 确定最后一个特征情况下的所属类别，返回的是个数最大的那个类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount.setdefault(vote, 0)  #构造list = [Yes:1,No:2]，统计每个类的个数
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda i:i[1], reverse=True)  #按照类的个数降序排列
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet] # 提取最后一列的类别信息

    # 类别相同，停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 判断是否遍历完所有的特征,是则返回个数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 按照信息增益最高选择分类特征属性
    bestFeat = chooseBestFeatureToSplit(dataSet) #分类编号
    bestFeatLabel = labels[bestFeat]  #该特征的label
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat]) #移除该label

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # 确定最佳属性下的分支
    for value in uniqueVals:
        subLabels = labels[:]  #子属性集合
        ## 构建数据的子集合，并进行递归!!
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# 构造待分类数据集
def createData():
    Data = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    Label = ['no sufacing', 'flippers']
    Data = [[0, 0, 0, 0, 0, 0, 'yes'],
            [1, 0, 1, 0, 0, 0, 'yes'],
            [1, 0, 0, 0, 0, 0, 'yes'],
            [0, 0, 1, 0, 0, 0, 'yes'],
            [2, 0, 0, 0, 0, 0, 'yes'],
            [0, 1, 0, 0, 1, 1, 'yes'],
            [1, 1, 0, 1, 1, 1, 'yes'],
            [1, 1, 0, 0, 1, 0, 'yes'],
            [1, 1, 1, 1, 1, 0, ' no'],
            [0, 2, 2, 0, 2, 1, ' no'],
            [2, 2, 2, 2, 2, 0, ' no'],
            [2, 0, 0, 2, 2, 1, ' no'],
            [0, 1, 0, 1, 0, 0, ' no'],
            [2, 1, 1, 1, 0, 0, ' no'],
            [1, 1, 0, 0, 1, 1, ' no'],
            [2, 0, 0, 2, 2, 0, ' no'],
            [0, 0, 1, 1, 1, 0, ' no']]
    Label = ['色泽','根蒂',"敲声","纹理","脐部","触感"]
    return Data, Label

# main
if __name__ == '__main__':
    Data, Label = createData()
    r = chooseBestFeatureToSplit(Data)
    #print(r)
    myTree = createTree(Data, Label)
    print(myTree)

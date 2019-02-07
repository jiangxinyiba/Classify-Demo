# Decision-Tree-Demo
A python demo which use ID3 algorithm for classification
# RandomForest-demo
采用集成学习中的Bagging思想改进决策树【之后才发现这个其实就是随机森林】
这里分别采用以下方法：
1.一般决策树【scikit-learn中的DecisionTreeClassifier方法】
2.采用集成学习中的Bagging思想改进上面的决策树
3.随机森林【scikit-learn中的RandomForestClassifier方法】
4.极限树【scikit-learn中的ExtraTreesClassifier方法】
其中bagging方法通过样本重采样生成多组子模型分别进行决策树分类器训练，其中子模型个数和采样率可以设置
最后通过投票法确定集成模型的分类结果，数据集分别用了西瓜数据集3.0和make_blobs数据集进行分析。
代码最后使用Graphviz生成了决策树结构，具体如pdf和dot文件所示，如需运行此功能，需要安装Graphviz等模块，
可以参照https://blog.csdn.net/tina_ttl/article/details/51778551#%E9%97%AE%E9%A2%98%E5%87%BA%E7%8E%B0 进行安装配置
# Logistic回归算法
根据机器学习实战第5章“Logistic回归”，基于附属的代码修改后可以在python3下正常运行，
修改说明如下：
1.部分代码按照python3的逻辑进行修改
2.在colicTest函数中，基于当前数据的各列特征之间范围差距过大的问题，导致math.exp()无法求解，对训练数据进行归一化处理，其中归一化采用scikit中的preprocess库实现。
3.各梯度算法函数说明。
  gradAscent是梯度上升算法；
  stocGradAscent0是普通的随机梯度上升算法，采用增量式更新权值，训练了所有的训练数据一遍；
  stocGradAscent1是改进的随机梯度上升算法，进行了numIter次迭代，每次都是随机选取样本增量式更新权值,本函数用于马疝分类问题；
  stocGradAscent1_gai改进的随机梯度上升算法，修改后可以展示每个权值的更新情况，本函数用于testSet.txt的数据分类展示。

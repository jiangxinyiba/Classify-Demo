# Decision-Tree-Demo
A python demo which use ID3 algorithm for classification
# Bagging-DecisionTree demo
采用集成学习中的Bagging思想改进决策树
其中决策树采用了scikit-learn中的DecisionTreeClassifier方法进行处理
bagging方法通过样本重采样生成多组子模型分别进行决策树分类器训练，其中子模型个数和采样率可以设置
最后通过投票法确定集成模型的分类结果
代码最后使用Graphviz生成了决策树结构，具体如pdf和dot文件所示，如需运行此功能，需要安装Graphviz等模块，
可以参照https://blog.csdn.net/tina_ttl/article/details/51778551#%E9%97%AE%E9%A2%98%E5%87%BA%E7%8E%B0 进行安装配置
#

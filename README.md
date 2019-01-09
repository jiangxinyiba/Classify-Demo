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
#

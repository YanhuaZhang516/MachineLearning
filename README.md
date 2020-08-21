# MachineLearning(Germany+Chinese)
学习+面试准备
## 1. [the concept of Machine Learning 机器学习基本概念](https://github.com/YanhuaZhang516/MachineLearning/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%A6%82%E5%BF%B5.md)
- [x] 8.15

## 2. [Lineare Modelle 线性模型](https://github.com/YanhuaZhang516/MachineLearning/blob/master/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B.pdf)
- [x] 8.18
  - Lineare Regression 线性回归
    - multivariate Regression
    - Optimierung problem
      - Normalgleichung/normal equation 矩阵方程法[normal equation 和梯度下降法的比较](https://blog.csdn.net/Artprog/article/details/51172025)
      - [为什么样本方差分母是n-1(无偏估计)](https://www.zhihu.com/question/20099757)
    - Regressionkoeffizienten
    - Basisfunktionen-Modelle 线性回归模型
    - Regularisierung 正则化
  - Ridge Regression 脊回归
  - Logistische Regression 逻辑回归
  - Verlustfunktion
  - Zielfunktionen
  
  ## 3. [Support-Vector-Machines 支持向量机](https://github.com/YanhuaZhang516/MachineLearning/blob/master/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA.pdf)
  - [x] 8.20
  
  感知机模型的基本思想是：先随机选择一个超平面对样本点进行划分，然后当一个实例点被误分类，即位于分类超平面错误的一侧时，则调整w和b，使分类超平面向该误分类点的一侧移动，直至超平面越过该误分类点为止。所以，如果给的初始值不同，则最后得到的分割超平面wx+b=0也可能不同，即感知机模型的分割超平面可能存在很多个。
  
  SVM模型：不仅要让样本点被分割超平面分开，还希望那些离分割超平面最近的点到分割超平面的距离最小。（选出最佳的分割面）
  
  注意：感知机模型有两个缺点，即当训练数据集线性不可分时，感知机的学习算法不收敛，迭代过程会发生震荡。另外，感知机模型仅适用于二分类问题，在现实应用中存在一定局限性。
  
- Maximal-Margin-Klassifikation
- Entscheidungsfunktion
- Optimierungsproblem
- Grenzfälle(Schlupvariablen)
- Soft-Margin-Klassifikation
- Robuste Klassifikation
- Optimierung
  - Primales Problem
  - Duales Problem
- KKT
- Duales SVM-Problem
  - [SMO算法](https://zhuanlan.zhihu.com/p/29212107)
  - [如何求得b值](https://weread.qq.com/web/reader/bc532d1071845519bc5b2a1k6c8328f022d6c8349cc72d5)
  - Interpretation der dualen Loesung
- [NichtLineare SVMs](https://www.jianshu.com/p/1b6e31c55f50)
  - Interpretation der nichtlinearen SVM
- Duales Problem mit Kernel

通过引入松弛变量，线性支持向量机可以有效解决数据集中带有异常点的情况。但实际中，很多数据可能并不只是带有异常点这么简单，而是完全非线性可分的,引入核函数
  - Kernel-Funktionen
- SVM-Regression
- Einfluss der SVM-Hyperparameter

## 4. Bäume und Ensembles 决策树和基模型
- [ ] 8.21

- 决策树

决策树（Decision Tree）是一种树状结构模型，可以进行基本的分类与回归，另外它也是后面要讲的集成方法经常采用的基模型。决策树主要涉及三要素：特征选择，决策树的生成和决策树的剪枝。
- Inhomogenitätsmasse 特征取值的非均匀性
- CART-Algorithmus 
- Regression-tree
  - CART for regression
- 集成方法(Ensemble-lernen)
  - Bias/Varianz-Zerlegung
- Bagging

Bagging模型的核心思想是每次同类别、彼此之间无强关联的基学习器，以均等投票机制进行基学习器的组合。
  - Bagging with decision tree
  - Random Forest 随机森林
- Boosting

与Bagging模型不同，Boosting模型的各个基学习器之间存在强关联，即后面的基学习器是建立在它前面的基学习器的基础之上的。Boosting模型的代表是提升算法（AdaBoost），更具体的是提升树（Boosting Tree），而提升树比较优秀的实现模型是梯度提升树（GBDT和XGBoost)。
- AdaBoost
  - AdaBoost mit Baumstümpfen
- Stufenweise additive Modellierung 逐级建模
- Boosting als stufenweise Konstruktion
- Gradient-Boosting
- Ensemble-Verfahren
- Random Forests und Boosting
 





## 5. Training und Bewertung
- [ ] 8.22
## 6. Unüberwachtes Lernen
- [ ] 8.22

#  DeepLearing-part(English)
## 参考资料：
- [CNN in computer vision (standford)](https://cs231n.github.io/convolutional-networks/)
- [deep learning-CNN,RNN(standford)](https://stanford.edu/~shervine/teaching/cs-230/)
## 7. Neuronale Netze
- [ ] 8.23
## 8. Faltende Neuronale Netze(CNNs)
- [ ] 8.24
## 9. CNN-Architekturen und -Anwendungen
- [ ] 8.24
## 10. Rekurrente Neuronale Netze
- [ ] 8.25

  
  

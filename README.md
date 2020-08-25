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
- Duales SVM-Problem[为什么要转化为对偶问题](https://www.zhihu.com/question/36694952)

要注意用拉格朗日对偶并没有改变最优解，而是改变了算法复杂度：在原问题下，求解算法的复杂度与样本维度（等于权值w的维度）有关；而在对偶问题下，求解算法的复杂度与样本数量（等于拉格朗日算子a的数量）有关。因此，如果你是做线性分类，且样本维度低于样本数量的话，在原问题下求解就好了，Liblinear之类的线性SVM默认都是这样做的；但如果你是做非线性分类，那就会涉及到升维（比如使用高斯核做核函数，其实是将样本升到无穷维），升维后的样本维度往往会远大于样本数量，此时显然在对偶问题下求解会更好。
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
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
- [ ] 8.25

- 决策树

决策树（Decision Tree）是一种树状结构模型，可以进行基本的分类与回归，另外它也是后面要讲的集成方法经常采用的基模型。决策树主要涉及三要素：特征选择，决策树的生成和决策树的剪枝。
- Inhomogenitätsmasse 特征取值的非均匀性
  - kreuzentropie(Devianz):
  在信息论或概率统计中，用熵度量随机变量的不确定性。熵值越大，随机变量的不确定性就越大。
  - Gini-index:
  基尼系数（Gini）可以用来度量任何不均匀分布，且介于0~1之间的数（0指完全相等，1指完全不相等）。分类度量时，总体包含的类别越杂乱，基尼系数就越大（与熵的概念相似）。
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
- [ ] 8.25

- Fehlermasse
不同机器学习任务往往需要使用不同的评估指标，下面分别介绍机器学习中回归模型、分类模型和聚类模型的评估指标
- Bias/Varianz
  - 偏差，方差分解 
  - 偏差，方差与模型复杂度的关系
- Modell-Validierung
- lernkurven
- Wahl der Hyperparameter
- Kreuzvalidierung(cross validation)
- Bootstrapping
- Fehler-Schätzung durch Resampling
  - KCV-Schätzung
  - Bootstrap-Schätzung
 - Binäre Klassifikation 分裂模型的评估指标
  - Accuracy 准确率（ACC=TP+TF/P+N）
  
  准确率就是用来衡量模型对数据集中**样本预测正确**（包括positive/我想要的，negative/我不想要的）的比例，即等于所有预测正确的样本数目与所有参加预测的样本总数目的比
  - Precision 精度（PPV= TP/TP+FP）
 
 精度指的是所有**预测为正例**的样本（TP+FP）中**真正为正例的样本（TP）**的比率。一般来说，就是你找出的这些信息中**真正是我想要的有多少**，又叫“查准率”。（常用于购物推荐）
  
  - sensitivity,recall 召回率 True postive rate (TPR=TP/TP+FN)
  
  召回率指的是所有为**正例**的样本（TP+FN）中**真的正例（TP）**的比率，用来评判你有没有把样本中所有的真的正例全部找出来，所以又叫“查全率”。通俗地讲，就是你有没有把所有我感兴趣的都给找出来（注意：查全率在犯罪检索等行为中可能比较重要。）
  - 相对应的关系
    - Precision-Recall-kurve: F1 score
    - Recevier-Operating-Charakteristik(ROC curve)
  - Mehrklassen Klassifikation
    - Warheit matrix
  - 分类检测考量的具体操作
  1. Merkmale visualisieren 特征值分布视觉化 （Histogram or Gauss-modell）
  2. Vorverarbeitung 数据预处理
  3. Daten aufteilen 数据分类（Traningsdaten, Testdaten）
  4.1 Referenz-Klassifikator 
    - 线性分类器
    - 非线性： AUC曲线
  4.2 Lernkurven (Fehlerklassifikationsrate)
  5. Hyperparameter bestimmen 不同模型下，调整参数对误差的影响
    - SVM Gauß-Kernel
    - Random Forest
  6.1 Kandidaten auswählen:
  算出不同模型下通过k-fold cross validation算出的 Validierungsfehler 
  6.2. 绘制出ROC曲线
  7. Arbeitspunkt festlegen
  8. Finale Entscheidung
  
 从多维度判断：
  - 1. Testfehler
  - 2. KCV fehler
  - 3. AUC(area under curve)
  - 4. Modellgröße
  - 5. Auswertezeit
  


## 6. Unüberwachtes Lernen
- [ ] 8.25
- K-means Clustering
  - Algorithmus
    - Abbruchkriterium und Cluster-Anzahl
    - Instabilität von K-means
    - Cluster-Analyse
  - Application:
    - Anomalie-Detektion
    - Dichte-Modelle
  - PCA
  - Manifold-Learning

#  DeepLearing-part(English)
## 参考资料：
- [CNN in computer vision (standford)](https://cs231n.github.io/convolutional-networks/)
- [deep learning-CNN,RNN(standford)](https://stanford.edu/~shervine/teaching/cs-230/)
## 7. Neuronale Netze
- [x] 8.23
- 基本原理
  - 一层神经结构
  - 一个神经元
  - [ ] more details to do 前向传播和反向传播
  - [ ] to do [Backpropagation](https://cs231n.github.io/optimization-2/)
  
  基本原理：利用前向传播可以算出预测值，并且计算出其与理论值之间的均方误差。然后求和算出总均方误差。然后开始利用反向传播更新各个权重和偏置常数，目标是使最后输出的总均方误差最小，采用梯度下降法更新。
  
  - 常用激活算法
  - 优化算法 [Gradient descent optimization](https://ruder.io/optimizing-gradient-descent/index.html#gradientdescentvariants)
    - 梯度下降 Gradient descent
      - Steigst descent
        - 非自适应算法
          - Gradient descent
          - Momentum-Optimization
          - Nesterov accelerated gradient
        - 自适应算法
          - RMSProp
          - AdaGrad
          - Adam 
      - 对于更高维度
        - 共轭 CG
        - Levenberg-Marquardt(LM): Quasi-Newton for quadrate error
        - BFGS
 
## 8. Faltende Neuronale Netze(CNNs)
- [ ] 8.24
- LeNet
  - Bild-Klassification
  - Faltende neuronale Netze  
    - convolutional layer
    - Pooling layer
    - fully-connected layer
- AlexNet
- Bildfilterung
  - hubel & Wiesel
  - lineare Bildfilterung
    - Gaussian filter
    - Sobel-canten detection
- Eingangsbild
    - Gray frame (1 ebene)
    - RGB frame (3 ebene)
    - RGB+depth frame(4 ebene)
    - Audio signal (sequence)
- parameter kernel
- CONV layer
- Stride & Padding
- Pooling layer
- characters of CNNs
  - characters
  - Problem in the training
- [梯度消失问题 Vanishing Gradient Problem](https://cs231n.github.io/neural-networks-1/#actfun)

当激活函数是sigmoid 函数时，当input的值很大/小时，它的值无限逼近0或1，Gradient几乎为0。因此在反向传播计算时，会导致计算值为0，从而导致这部分信息丧失。
When the neuron’s activation saturates at either tail of 0 or 1, the gradient at these regions is almost zero. Recall that during backpropagation, this (local) gradient will be multiplied to the gradient of this gate’s output for the whole objective. Therefore, if the local gradient is very small, it will effectively “kill” the gradient and almost no signal will flow through the neuron to its weights and recursively to its data. 

- 问题1：梯度消失
  - 改变Activation function(Sättigungsfrei Aktivierung)
  Ohne Begrenzung; schnelles Training, auch wenn nicht glatt
    - ReLU =max(0,z)
    - Leaky ReLU = max(\alhpa z,z)
    - Exponential-linear unit(ELU)
  - Weight的初始化，不要取过大的值
  - Batch Normalization
  
  激活函数的输入分布保持在一个稳定状态来尽可能避免它们陷入梯度饱和区。
- 问题2：参数 W 数量过多  
  - Regularisierung tiefer Netze
    - weight decay,early stopping
  - Dropout: While training, dropout is implemented by only keeping a neuron active with some probability p (a hyperparameter), or setting it to zero otherwise.
- Training tiefer Netze
  - MNIST
## 9. CNN-Architekturen und -Anwendungen
- [x] 8.24

Convolutional Neural Networks are very similar to ordinary Neural Networks from the previous chapter: they are made up of neurons that have learnable weights and biases. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. The whole network still expresses a single differentiable score function: from the raw image pixels on one end to class scores at the other. And they still have a loss function (e.g. SVM/Softmax) on the last (fully-connected) layer and all the tips/tricks we developed for learning regular Neural Networks still apply.

 ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.

- LeNet and AlexNet
- VGGNeet
  - VGGNet-Varianten
- GoogLeNet and Inception
  - Inception-module
  - 1* 1 Conv
- ResNet
  - Residuen verbindung
- Vorbereiten der Trainingsdaten
- Effizientes Labeling
- Mehrfachnutzung von Daten
  - Data augmentation
- Transfer-learning
- Application:
  - Hardware/Softeware for deep learning
  - Localization, Segmentation, Detection
  - Object- detection:
    - Fenster
    - R-CNN
    - Fast R-CNN
    - Faster R-CNN
    - Mask R-CNN
  - single-shot-detection
  -Frameworks for object-detection
  
## 10. Rekurrente Neuronale Netze
- [ ] 8.24
- RNN Base cell
- Abrollen von RNNs
- RNN-configuration
- Training of RNNs
- components of Base cell
- Gradient float
- Long Short-Term memory
- LSTM cell
- Encoder-decoder netze


  
  

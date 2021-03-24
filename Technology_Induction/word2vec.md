在语言模型中（包括word2vec），由于参数空间过大、数据过于稀疏，计算不方便，所以只考虑近邻n个词对其影响，以简化计算。即为N-gram模型，N为超参数。

word2vec的用途：如何把词转化为向量。

神经网络：

​	目标是层层之间的参数、还有优化后的输入向量。如何优化：求最大似然函数，即为目标函数最大值（词出现在该环境中的最大可能）。用提度上升求最大似然函数。

先更新的是层层参数（最大似然函数对参数求偏导），然后再更新输入向量（最大似然函数对投影层和向量求偏导，直接将和向量的更新量整个应用到每个单词词向量上去）。因为要最好的层层参数，所以原料要好，所以顺便更新输入词向量（副产品），但是副产品我们刚好需要的。

​	输入层：上下文单词的onehot编码，先随机初始化矩阵W，one-hot和W相乘为输入向量，然后不断迭代优化。

​	映射层：把输入向量首尾相加

​	输出层：



CBOW：continuous-bag-of-words



需要定义loss function（一般为交叉熵代价函数），采用梯度下降算法更新W。训练完毕后，输入层的每个单词与矩阵W相乘得到的向量的就是我们想要的词向量（word embedding），也就是说，任何一个单词的onehot乘以这个矩阵都将得到自己的词向量。





但是使用One-Hot Encoder有以下问题。一方面，城市编码是随机的，向量之间相互独立，看不出城市之间可能存在的关联关系。其次，向量维度的大小取决于语料库中字词的多少。如果将世界所有城市名称对应的向量合为一个矩阵的话，那这个矩阵过于稀疏，并且会造成维度灾难。 使用Vector Representations可以有效解决这个问题。Word2Vec可以将One-Hot Encoder转化为低维度的连续值，也就是稠密向量，并且其中意思相近的词将被映射到向量空间中相近的位置。 



用TensorFlow来实现Word2Vec。

**两次训练出来的词向量不一样是很正常的**，因为**在你给模型指定的任务中，并没有对每个词向量的绝对位置的约束**，这也就导致了每次生成的词向量不一样。<https://www.zhihu.com/question/57144800/answer/681943645>



###### 训练word2vec词向量时，内存不足如何解决？可以使用生成器，一点一点的从文档中拿出数据做训练，就算你训练文本再多，也OK的。



基于神经概率模型的word2vec：输入层到映射层是拼接的；有隐藏层有映射层；论文中的word2vec在输入层和隐藏层之间是有权重矩阵W的，one-hot成这个训练好的W得到200维词向量，它的输出层是神经网络，输出层是线性的；。

基于CBOW的word2vec：输入层到映射层是累加求和的；无隐藏层有映射层，我们用的是用哈夫曼树改进过的word2vec，输出层是树型的，。









收藏：

<https://blog.csdn.net/sinat_34080511/article/details/69665023>利用word2vec进行文档分类。

<https://zhuanlan.zhihu.com/p/26306795>

<https://zhuanlan.zhihu.com/p/72664857>

<https://blog.csdn.net/itplus/article/details/37969519>大神回答

<https://www.cnblogs.com/pinard/p/7160330.html>大神回答

<http://www.aboutyun.com/thread-24275-1-1.html>

<https://blog.csdn.net/zhaoxinfan/article/details/11069485>

<https://www.cnblogs.com/coshaho/p/9571090.html>

<https://blog.csdn.net/ldx19980108/article/details/78587784>哈夫曼树

<https://985359995.iteye.com/blog/2436025>词向量相加




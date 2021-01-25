# Language Model

## Reference

+ [1] https://docs.google.com/presentation/d/1h51R4pMeZkS_CCdU0taBkQucUnPeyucQX433lsw8bVk/edit#slide=id.g1ea65cc3ed_1_578
+ LHY DeepLearning Y2017 4
+ 2003年，Bengio等人发表了一篇开创性的文章：A neural probabilistic language model
+ 斯坦福大学自然语言处理第四课“语言模型（Language Modeling）”
	+ http://blog.csdn.net/xiaokang06/article/details/17965965
+ https://blog.csdn.net/shiwei1003462571/article/details/43482881
+ 如何使用N-gram语言模型来进行篇章单元分类？
	+ https://www.imooc.com/article/20929

## Define
+ Estimated the probability of word sequence
	+ Word sequence: $w_1, w_2, ..., w_n$
	+ $P(w_1, w_2, ..., w_n)$

## Application
+ speech recognition
	+ Difference word sequence can have the same pronunciation
	+  'recognize speech' or 'wreck a nice beach'
		+ if(P(ecognize speech) > P(recognize speech or wreck a nice beach)) 
		+ output is 'revognize speech'
+ sentence generation

# Ｎ-Gram language model
+ How to estimate $P(w_1, w_2, ..., w_n)$ ?
+ collect a large amount of text data as training data
 	+ However, the word sequence $w_1, w_2, ..., w_n$ may not appear in the training data
        + In N-Gram Model : $P(w_1, w_2, ..., w_n) = P(w_1|START)P(w_2|w_1)......P(w_n|w_{n-1})$ 
        + $P(w_{i+1}|w_i)$ 可以从数据中估测出来
        	+ 例如 :  $P(beach|nice) = \frac{C(nice\ beach)}{C(nice)}$
+ Drawback
	+ the estimated probability is not accurate
		+ Especially when consider n-gram with large n,  the model is big and data is not sufficient 
+ Solution of Drawback(Data sparse)
	+ smoothing
	+ matrix Factroization(consider it as a NN)
		+ history
		+ vocabulary
		+ 矩阵中是零的值 由 history 和 vocabulary 乘出来
		+ softmax 转化为概率
		+ cross entropy

# NN-based language model
+ collect data
+ 根据前Ｎ个词去预测下一个词
+  Minimizing cross entorpy
+  RNN-based LM(观察之前所有的词和当前词，去预测下一个词)
	+  $P(w_1, w_2, ..., w_n)  = $ P(w_1) P(w_2|w_1) P(w_3|w_1,w_2)  P(w_4|w_1,w_2, w_3)  $ 
+ Advantage
	+ 参数远小于N-Gram LM
	+ **RNN-based LM** 需要的参数更少
+ output layer issue and solutions[1]
	+ issue 
		+ 
	+ solution:
		+ Sampling Method : 
			+ Noise contrastive estimation language model
			+ Randomly sample some words to suppress the probability
			+ Only part of weight s would be updated
		+ Softmax-based Method
			+ Hierachical Sofrmax
				+ How to define the word hierarchy?
					+ randomly generated tree
					+ existing linguistic resources, example:WordNet
					+ Hierarchical clustering
			+ Differentiated Softmax
			+ CNN-Softmax
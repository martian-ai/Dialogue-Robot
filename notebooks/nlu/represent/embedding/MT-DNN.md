
# MT-DNN
[Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/pdf/1901.11504.pdf)采用的思想运用多任务学习（Multi-tasks Learning MTL）的机制来进行模型的训练。不同于普通的BERT以及改进，都是在fine-tuning阶段才进行特定任务的学习，但是这篇论文认为应该在pre-training的时候就加入这些NLU的任务，pretraining的任务和MTL的任务应该是互补的，这样子可以得到更好的效果。

但是为什么要叫MT-DNN呢，因为这篇文章是在[原本的模型](http://cs.jhu.edu/~kevinduh/papers/liu15multitask.pdf)基础上改进的，这个模型采用的是使用两层的DNN作为shared layers，但是在这个论文中，使用的了BERT的模型输入和输出作为共享层。

模型取得了SOTA的效果，这也给了百度的ERNIE2.0提供了很大的启发，例如加入更多的任务作为预训练的objectives，以及如何有效地共同训练模型：每一轮中，mini-batch用来训练特定的任务，最后近乎是同时训练多任务的目标。

文中提出的两个句子的相关性的任务，可以使用pair wise 的rank做，能够得到更好的效果。
[repo](https://github.com/namisan/mt-dnn)

## 为什么要MT-DNN
现在的NLU任务中，主要有两种方法，一种是language model pretraining一种的multi-task learning，但是两者并不是单独独立的，需要把两者结合。
- **multi-task learning**  
  - 因为我们单一任务的数据是有限的，multi-task 提供了更多的有标签的训练数据
  - multi-task learning 起了一个regularization的作用，模型不会在特定任务上overfitting。使得学习到的模型更普遍适用
- **language model pretraining**
  - 语言模型的预训练提供了模型一个普遍适用的语言表示 using unlabeled data
- 更容易进行domain adaption，需要的数据更少


## Multi-Tasks
- Pretraining tasks：
和bert一样作为预训练
  - Mask Language Modeling
  - Next Sentence Prediction

- single sentence classification
- pairwise text classification
- text similarity
- relevance ranking

## MT-DNN model


![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/05219749ab65c7e531f2a02effbc9658.png)
- 输入层lexicon encoder：X=[CLS]+SENTENCE1+[SEP]+【SENTENCE2+[SEP]】和BERT一致，由word+segment+position encoding 组成，其结果表示为$l_1$
- 中间层Transformer encoder：中间层使用的是transformer的encoder，将输入进行编码，得到每个token对应一个embedding vector，其结果表示为$l_2$
- 任务层：
  - **single sentence classification**：使用的是[CLS] 的$l_2$输出作为分类层的输入，经过softmax，之后使用cross entropy作为loss。
  - **text similarity**：使用的是[CLS] 的$l_2$输出作为回归层的输入，直接输出，之后使用MSE作为loss。
  - **pairwise text classification**：模型不直接使用[CLS]，而是使用了Stochastic answer network（SAN）模型的输出层作为最后分类层。
    > TODO：[Stochastic Answer Networks for Machine Reading Comprehension](https://arxiv.org/pdf/1712.03556.pdf)
    ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/9167b38ebc64e5750469abe79829c46b.png)

    SAN的效果，可以看到，下面的这些Pairwise text classification 任务，使用ST-DNN（和bert相比仅是模型结构+SAN），的效果比BERT好。
    ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/fbe44b1a0a964ac14fcf7044ab64830f.png)

  - **relevance ranking**：两个句子相关性任务，拿QNLI这个数据集举例，虽然QNLI设计为2分类，但是这边使用的是rank的方式来训练，并且发现效果能够提升非常多。

    模型使用的是[CLS]的$l_2$输出$x$作为（Q,A）pair的context 输出，然后经过计算相关度$Rel(Q, A) $
    \begin{equation}
          Rel(Q, A) = g(W_{QNLI}^⊤ · x)
    \end{equation}
    其中$g(·)$为激活函数。
    对于$Q$的所有candidates $A$, 都计算一个相关度。
    我们有1个正样本$A^+$，|A|-1 个负样本$A^-$。我们希望所有正样本的似然概率最大
    \begin{equation}
    \begin{split}
          \prod_{(Q,A^+)} P_r(A^+|Q) =&  
          -\sum _{Q,A^+} P_r(A^+|Q) \\
          其中P_r(A^+|Q) &= \frac {e^{\gamma Rel(Q,A^+)}} {\sum _{A' \in A} e^{\gamma Rel(Q,A)}}
    \end{split}
    \end{equation}
    $\gamma =1$ 在这个模型中
    下面的这些QNLI的任务，使用ST-DNN（和bert相比仅是模型结构+SAN），和BERT相比，仅是将任务进行了重新的formalation了，效果得到了极大的提升。
    ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/fbe44b1a0a964ac14fcf7044ab64830f.png)

- 训练和使用
  和BERT一样，有两阶段，一阶段是pretraining 一阶段是finetuning。
  - 在pretraining阶段
    - 先用：MLM以及NSP进行lexicon encoder和Transformer encoder的初始化。
    - 然后在进行四个任务的Multi-task learning，轮流训练
  ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/f844e1d1e5c9b4d928d714f0dd67b4f0.png)
  - fine-tuning阶段和BERT一致。

## ref
[https://github.com/namisan/mt-dnn/blob/master/tutorials/Run\_Your\_Own\_Task\_in\_MT-DNN.ipynb](https://github.com/namisan/mt-dnn/blob/master/tutorials/Run_Your_Own_Task_in_MT-DNN.ipynb)
https://arxiv.org/pdf/1901.11504.pdf
https://fyubang.com/2019/05/23/mt-dnn/
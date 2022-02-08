# Markov Contiontion Random Field

## Reference
+ 李宏毅 PDF
+ https://blog.csdn.net/weixin_39999222/article/details/110518643
+ 李航 统计学习方法
+ 维特比算法 https://www.zhihu.com/question/294202922/answer/489485567 解释的很清楚

## 问题定义
+ $P(y_1,y_2,...,y_k | x_1, x_2,...,x_k) = \prod_{k=1}^{\ell} P\left(y_{k} \mid \mathbf{x}_{k}\right)$
+ 如果使用分类问题来处理
    + U(x，y) 被称为emissions或一元分数(unary scores)， 只是在第k个时间步给出x向量的标签y的分数，可以将其视为LSTM的第k个输出
    + Z(x) 通常称为配分函数(partition function)，将其视为归一化因子，因为想获得概率：每个不同标签的分数应该总和为1.可以将其视为softmax函数的分母
    + 使用exp 进行处理, 使用原因如下
        + 下溢:当我们用非常小的数字,我们得到一个较小的数量可能遭受下溢 （exp 最小值 为1，在输入大于0的情况下）
        + 非负输出:所有值都映射在0和+inf之间
        + 单调增加：这与argmax操作具有类似的效果
+ 经过上一步的函数和符号转化后变为
   + $\begin{aligned} P(y_1,y_2,...,y_k | x_1, x_2,...,x_k) &=\prod_{k=1}^{\ell} P\left(y_{k} \mid \mathbf{x}_{k}\right) \\ &=\prod_{k=1}^{\ell} \frac{\exp \left(U\left(\mathbf{x}_{k}, y_{k}\right)\right)}{Z\left(\mathbf{x}_{k}\right)} \\ &=\frac{\exp \left(\sum_{k=1}^{\ell} U\left(\mathbf{x}_{k}, y_{k}\right)\right)}{\prod_{k=1}^{\ell} Z\left(\mathbf{x}_{k}\right)} \end{aligned}$

## CRF 模型
+ 上一步未考虑标签之间的依赖关系， 现在添加新的可学习权重来模拟标签$y_k$跟随$y_{k + 1}$的可能性
+ 添加依赖关系后就构建成了线性链CRF模型
+ 为了做到这一点，将先前的概率乘以$P(y_{k + 1}| y_k)$，可以使用指数属性将其重写为unary scores U(x，y)加上可学习的transition scores T(y，y)
    + 之前的问题是 计算 $P(y_1,y_2,...,y_k | x_1, x_2,...,x_k) = \prod_{k=1}^{\ell} P\left(y_{k} \mid \mathbf{x}_{k}\right)$
    + 添加依赖关系后的 计算 $P(y_1,y_2,...,y_k | x_1, x_2,...,x_k) = \prod_{k=1}^{\ell} P\left(y_{k} \mid \mathbf{x}_{k}\right)  P\left(y_{k+1} \mid \mathbf{y}_{k}\right) $

+ 整体的公式如下
    + $P(y_1,y_2,...,y_k | x_1, x_2,...,x_k)=\frac{\exp \left(\sum_{k=1}^{\ell} U\left(\mathbf{x}_{k}, y_{k}\right)+\sum_{k=1}^{\ell-1} T\left(y_{k}, y_{k+1}\right)\right)}{Z(\mathbf{X})}$
    + 其中的配分函数计算较为复杂
        + $Z(\mathbf{X})=\sum_{y_{1}^{\prime}} \sum_{y_{2}^{\prime}} \cdots \sum_{y_{k}^{\prime}} \cdots \sum_{y_{\ell}^{\prime}} \exp \left(\sum_{k=1}^{\ell} U\left(\mathbf{x}_{k}, y_{k}^{\prime}\right)+\sum_{k=1 \atop=1}^{\ell-1} T\left(y_{k}^{\prime}, y_{k+1}^{\prime}\right)\right)$
        + 事实证明，计算Z(X)并非易事，因为有太多的嵌套循环！
        + 它是每个时间步标签集上所有可能组合的总和, **对标签集进行了$l!$计算**
            + **时间复杂度O(ℓ！| y |²)**
        + 幸运的是，可以利用循环依赖关系并使用动态编程来有效地计算它！
            + **执行此操作的算法称为前向算法或后向算法 - 取决于在序列上迭代的顺序。**

## 代码实现
+ 初始化
```
import torchfrom torch 
import nn
class CRF(nn.Module):
    “”“
    Linear-chain Conditional Random Field (CRF). 
    Args: nb_labels (int): number of labels in your tagset, including special symbols.
    bos_tag_id (int): integer representing the beginning of sentence symbol in your tagset. 
    eos_tag_id (int): integer representing the end of sentence symbol in your tagset. 
    batch_first (bool): Whether the first dimension represents the batch dimension.
    ”“”
    def __init__( self, nb_labels, bos_tag_id, eos_tag_id, batch_first=True ):         
        super().__init__() self.nb_labels = nb_labels 
        self.BOS_TAG_ID = bos_tag_id 
        self.EOS_TAG_ID = eos_tag_id 
        self.batch_first = batch_first 
        self.transitions = nn.Parameter(torch.empty(self.nb_labels, self.nb_labels)) 
        self.init_weights() 
    def init_weights(self): 
        # initialize transitions from a random uniform distribution between -0.1 and 0.1 nn.init.uniform_(self.transitions, -0.1, 0.1) 
        # enforce contraints (rows=from, columns=to) with a big negative number 
        # so exp(-10000) will tend to zero # no transitions allowed to the beginning of  sentence 
        self.transitions.data[:, self.BOS_TAG_ID] = -10000.0 
        # no transition alloed from the end of sentence 
        self.transitions.data[self.EOS_TAG_ID, :] = -10000.0
```

+ loss 计算
    + loss = $-\log (P(y_1,y_2,...,y_k | x_1, x_2,...,x_k))$
    $\begin{aligned}&=-\log \left(\frac{\exp \left(\sum_{k=1}^{\ell} U\left(\mathbf{x}_{k}, y_{k}\right)+\sum_{k=1}^{\ell-1} T\left(y_{k}, y_{k+1}\right)\right)}{Z(\mathbf{X})}\right) \\ &=\log (Z(\mathbf{X}))-\log \left(\exp \left(\sum_{k=1}^{\ell} U\left(\mathbf{x}_{k}, y_{k}\right)+\sum_{k=1}^{\ell-1} T\left(y_{k}, y_{k+1}\right)\right)\right) \\ &=\log (Z(\mathbf{X}))-\left(\sum_{k=1}^{\ell} U\left(\mathbf{x}_{k}, y_{k}\right)+\sum_{k=1}^{\ell-1} T\left(y_{k}, y_{k+1}\right)\right) \\ &=Z_{\log }(\mathbf{X})-\left(\sum_{k=1}^{\ell} U\left(\mathbf{x}_{k}, y_{k}\right)+\sum_{k=1}^{\ell-1} T\left(y_{k}, y_{k+1}\right)\right) \end{aligned}$
    + 将mask 传入，避免无谓计算
        + 我们将mask矩阵传递给这些方法，以便它们可以忽略与pad符号相关的计算。为完整起见，mask矩阵看起来像：

        ![20220130193825](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220130193825.png)

```
def forward(self, emissions, tags, mask=None): 
    """
    Compute the negative log-likelihood. See `log_likelihood` method.
    """ 
    nll = -self.log_likelihood(emissions, tags, mask=mask) return nll
    
def log_likelihood(self, emissions, tags, mask=None): 
    """
    Compute the probability of a sequence of tags given a sequence of emissions scores.

    Args: 
        emissions (torch.Tensor): 
            Sequence of emissions for each label. 
            Shape of (batch_size, seq_len, nb_labels) if batch_first is True, 
            (seq_len, batch_size, nb_labels) otherwise. 
        tags (torch.LongTensor): 
            Sequence of labels. Shape of (batch_size, seq_len) if batch_first is True, (seq_len, batch_size) otherwise. 
        mask (torch.FloatTensor, optional): 
            Tensor representing valid positions. 
            If None, all positions are considered valid. 
            Shape of (batch_size, seq_len) if batch_first is True, 
            (seq_len, batch_size) otherwise. 
     Returns: 
        torch.Tensor: 
            the log-likelihoods for each sequence in the batch. Shape of (batch_size,) 
    """ 
    
    # fix tensors order by setting batch as the first dimension 
    if not self.batch_first: 
        emissions = emissions.transpose(0, 1) 
        tags = tags.transpose(0, 1) 
    if mask is None: 
        mask = torch.ones(emissions.shape[:2], dtype=torch.float) 
    scores = self._compute_scores(emissions, tags, mask=mask) 
    partition = self._compute_log_partition(emissions, mask=mask) 
    return torch.sum(scores - partition)
```

# LSTM-DSSM
+ Palangi, Hamid, et al. "Semantic modelling with long-short-term memory for information retrieval." arXiv preprint arXiv:1412.6629 2014.
+ 针对 CNN-DSSM 无法捕获较远距离上下文特征的缺点，有人提出了用LSTM-DSSM（Long-Short-Term Memory）来解决该问题。
+ 模型结构
  + LSTM 改造
    + LSTM-DSSM 其实用的是 LSTM 的一个变种——加入了peephole的 LSTM
    + ![20210814165533](https://i.loli.net/2021/08/14/dMnvOyzkF2H3gmQ.png)
    + 换一种图的表示
      + ![20210814165619](https://i.loli.net/2021/08/14/C3uspKHi5OJyWlM.png)
    + 这里三条黑线就是所谓的 peephole，传统的 LSTM 中遗忘门、输入门和输出门只用了 h(t-1) 和 xt 来控制门缝的大小，peephole 的意思是说不但要考虑 h(t-1) 和 xt，也要考虑 Ct-1 和 Ct，其中遗忘门和输入门考虑了 Ct-1，而输出门考虑了 Ct。总体来说需要考虑的信息更丰富了。
  + 完整结构
    + ![20210814165719](https://i.loli.net/2021/08/14/HA8NKe3Ixc5lvFU.png)  
    + 红色的部分可以清晰的看到残差传递的方向。

# Reference
+ Gers, Felix A., Schraudolph, Nicol N., and Schmidhuber, J¨urgen. Learning precise timing with lstm recurrent networks. J. Mach. Learn. Res., 3:115–143, March 2003.
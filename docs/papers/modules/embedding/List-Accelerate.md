# BERT Accelerate

+ Reducing BERT Pre-Training Time from 3 Days to 76 Minutes
  + https://www.jiqizhixin.com/articles/2019-04-03-7
  + https://arxiv.org/abs/1904.00962
  + batch size 调整
    + 问题: 大批量训练会加速训练过程，但是会产生 泛化误差(generalization gap)
      + 原始的BERT 直接拓展 batch size 效果并不好
    + 为了解决这个问题，Google Brain 提出了一种新的优化器 LAMB，在不损害准确率的情况下讲 batch_size 拓展到了65536
    + LAMB 是一款通用优化器，它适用于小批量和大批量，且除了学习率以外其他超参数均无需调整
    + 基线 BERT-Large 模型的预训练需要 100 万次迭代，而 LAMB 使用 65536/32768 的批量大小，仅需 8599 次迭代。研究者将批量大小扩展到 TPUv3 pod 的内存极限，在 76 分钟内完成了 BERT 的训练
    + LAMB
      + 自适应元素级更新(adaptive element-wise updating)
      + 逐层修正(layer-wise correction)
  + TPU v3 上运行

### 知乎团队 cubert

+ https://github.com/zhihu/cuBERT?spm=a2c6h.12873639.0.0.40c426f1Ez45Id
+ Highly customized and optimized BERT inference directly on NVIDIA (CUDA, CUBLAS) or Intel MKL, *without* tensorflow and its framework overhead.
+ **ONLY** BERT (Transformer) is supported.

### 基于bert 神马搜索在线预测性能提升

+ https://developer.aliyun.com/article/714552
+ 基于tensorflow 优化

  + 内存到显存的参数传输
  + Transformer 计算

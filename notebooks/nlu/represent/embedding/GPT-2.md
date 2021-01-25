
# GPT-2
在GPT-1刚发布不久之后，马上被BERT 霸榜了，openAI 于是紧接着发布了[GPT-2]((https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf))，意在无监督数据的情况下，实现zero-shot任务表现最好。

模型结构等都没有什么区别，主要的改进就是数据量足够大，模型足够大。能够达到很好的NLG效果。see tutorial：http://jalammar.github.io/illustrated-gpt2/
## Tips

+ https://www.cnblogs.com/robert-dlut/p/9824346.html

+ GPT = Transformer + UML-Fit
+ GPT-2 = GPT + Reddit + GPUs
+ OpneAI 2018
+ Improving Language Understanding by Generative Pre-Training
+ 提出了一种基于半监督进行语言理解的方法
  - 使用无监督的方式学习一个深度语言模型
  - 使用监督的方式将这些参数调整到目标任务上

+ GPT-2 predict next word
+ https://blog.floydhub.com/gpt2/
+ ![](https://paper-attachments.dropbox.com/s_972195A84441142620E4C92312EA63C9665C3A86AFFD1D713034FA568ADFC5F9_1555424144125_openai-transformer-language-modeling.png)

## Unsupervised-Learning

![](https://img2018.cnblogs.com/blog/670089/201810/670089-20181021105844156-2101267400.png)

## Supervised-Learning

+ 再具体NLP任务有监督微调时，与**ELMo当成特征的做法不同**，OpenAI GPT不需要再重新对任务构建新的模型结构，而是直接在transformer这个语言模型上的最后一层接上softmax作为任务输出层，然后再对这整个模型进行微调。额外发现，如果使用语言模型作为辅助任务，能够提升有监督模型的泛化能力，并且能够加速收敛

  ![](https://img2018.cnblogs.com/blog/670089/201810/670089-20181021105844634-618425800.png)

## Task specific input transformation

![](https://img2018.cnblogs.com/blog/670089/201810/670089-20181021105845000-829413930.png)
python-topic-model
==================

Implementations of various topic models written in Python. Note that some of the implementations (the models with MCMC) are extremely slow. I do not recommend to use it for large scale datasets.

Current implementations
-----------------------

* Latent Dirichlet allocation
  * [Collapsed Gibbs sampling](http://nbviewer.jupyter.org/github/arongdari/python-topic-model/blob/master/notebook/LDA_example.ipynb)
  * [Variational inference](http://nbviewer.jupyter.org/github/arongdari/python-topic-model/blob/master/notebook/LDA_example.ipynb)
* Collaborative topic model
  * Variational inference
* Relational topic model (VI)
  * [Exponential link function](http://nbviewer.jupyter.org/github/arongdari/python-topic-model/blob/master/notebook/RelationalTopicModel_example.ipynb)
* [Author-Topic model](http://nbviewer.jupyter.org/github/arongdari/python-topic-model/blob/master/notebook/AuthorTopicModel_example.ipynb)
* [HMM-LDA](http://nbviewer.jupyter.org/github/arongdari/python-topic-model/blob/master/notebook/HMM_LDA_example.ipynb)
* Discrete infinite logistic normal (DILN)
  * Variational inference
* Supervised topic model
  * [Stochastic (Gibbs) EM](http://nbviewer.jupyter.org/github/arongdari/python-topic-model/blob/master/notebook/SupervisedTopicModel_example.ipynb)
  * Variational inference
* Hierarchical Dirichlet process
  * Collapsed Gibbs sampling
* Hierarchical Dirichlet scaling process

* Topic Balance
  * Topic Balancing with Additive Regularization of Topic Models
* Hierarchical Topic Model
  * CluHTM - Semantic Hierarchical Topic Modeling based on CluWords


### ACL 2020
+ Short Text Topic Modeling with Topic Distribution Quantization and Negative Sampling Decoder.
+ Sparse Parallel Training of Hierarchical Dirichlet Process Topic Models.
+ Response Selection for Multi-Party Conversations with Dynamic Topic Tracking.
+ Neural Topic Modeling with Cycle-Consistent Adversarial Training.
+ Improving Neural Topic Models using Knowledge Distillation.
+ Friendly Topic Assistant for Transformer Based Abstractive Summarization.
+ Neural Topic Modeling by Incorporating Document Relationship Graph.
+ Tired of Topic Models? Clusters of Pretrained Word Embeddings Make for Fast and Good Topics too!.
+ tBERT


### 相关产品
+ 腾讯Angel
+ 微软 LightLDA
+ AliasLDA
+ F+LDA
+ WarpLDA
+ 百度 Fumalia
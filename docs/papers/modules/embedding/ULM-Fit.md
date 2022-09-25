# Summary of Universal Language Model Fine-tuning

## Reference

+ [fast.ai NLP](<http://nlp.fast.ai/category/classification.html>)
  + Classification
  + Model Zoo
  + Seq2Seq
  + Sequence Labeling
+ [Universal Language Model Fine-tuning for text Classification](<http://nlp.fast.ai/category/classification.html>)
+ fastai library provide module necessary to train and use UMLFIT

+ [Pre-train Model]([available here](http://files.fast.ai/models/wt103/))

## Abstract

## Introduction

+ word2vec just in first layer
+ Concatenate embeddings derived from other tasks with the input at different layers
  + still train main task model from scatch
  + Treat  pretrained embedding as fixed parameters
+ ULMFiT
  + The same 3-layer LSTM architecture with the same hyperparameters and no additions other than tuned dropout hyperparameters
+ IMDb
  + 具体的指标情况还需要再看

## Related Work

### Transformer Learning in CV

### Hypercolumns

### Multi-task Learning

### Fine-tuning



### Universial Language Model Fine-tuning

+ inductive transfer learning setting for NLP(使用其他任务提到当前任务性能)
  + Language Model in NLP ～ Image Net in CV
+ General-domain LM Pretraining
  + Wikitext-103
+ Target task LM fine-tuning
+ Target task Classifier fine-tuning

### Experiments

### Analysis
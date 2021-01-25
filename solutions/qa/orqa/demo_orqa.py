# 1. 当前使用的是三体小说数据做demo
# 使用es插入数据

# TODO 多文档阅读理解 
import os,sys
sys.path.append('../..')

from elastic_search import es_search


def orqa_nlu(query, history):
	"""
	section : nlu
	a. 分词/词性/实体
	b. 领域分类/意图识别（结合上下文，暂时投票）
	c. 情感判断（结合上下文，暂时投票）
	"""
	return {'query':query}

def orqa_search(nlu_results):
	"""
	section : search 
	a. 问句改写
		1. pass
	b. 召回
		1. 关键词召回
		2. 向量召回（改写前/后）
	c. 排序
		1. 使用selector 模块，结合对话历史，用户画像，系统设定
		2. 
	"""
	return es_search(nlu_results['query'])


def orqa_interceptor(results_search):
	"""
	section : interceptor
	"""
	pass

if __name__ == "__main__":
	history = ['','']
	query = '云天明送给程心的行星叫什么名字'

	results_nlu = orqa_nlu(query, history)
	results_search = orqa_search(results_nlu)
	print(results_search)

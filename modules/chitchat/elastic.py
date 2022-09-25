"""

es 相关构建情况
1. 本地mac 现有闲聊语聊 index 为 chitchat-v1

ToDo
es bulk 方式添加
es 其他函数整理
"""

import sys, os
sys.path.append('../..')
from tools.elastic.chitchat import es_search, es_insert
from tools.formator.chitchat import get_data, line_chunking

if __name__ == "__main__":
	"""
	添加
	"""
	# path = '../../resources/corpus/chitchat/format_v3.txt'
	# lines = get_data(path)
	# print(len(lines))
	# paras = line_chunking(lines)
	# print(len(paras))
	# print(paras[0])
	# print(paras[-1])
		
	# es_insert(lines, 'chitchat-v1')
	"""
	查询
	"""
	print(es_search('天王盖地虎', 3, 'chitchat-v1'))
	"""
	# 删除
	# """
	# #es_delete(ES_TEST_INDEX)
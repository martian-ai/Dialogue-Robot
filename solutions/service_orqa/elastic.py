"""

仅放到 solutions 中，不放到moduels中

es 相关构建情况
1. 本地mac 三体小说 index 为 document-threebody

ToDo
es bulk 方式添加
es 其他函数整理
"""

import sys, os
sys.path.append('../..')
from tools.elastic.document import es_search, es_insert
from tools.formator.document import get_data, paragraph_chunking

if __name__ == "__main__":
	"""
	添加
	"""
	# path = '../../resources/corpus/document/三体.txt'
	# lines = get_data(path)
	# paras = paragraph_chunking(lines, 384)
	# es_insert(paras, 'document-threebody')
	"""
	查询
	"""
	print(es_search('云天明送给程心的星球叫什么名字', 3, 'document-threebody'))
	"""
	删除
	"""
	#es_delete(ES_TEST_INDEX)
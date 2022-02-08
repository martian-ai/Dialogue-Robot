
import sys, os
sys.path.append('../..')

from modules.basic_search.recall.function.recaller import EsRecaller
from modules.structure_document.document import get_data

recaller = EsRecaller(ip='10.9.227.9', port='8200')

if __name__ == "__main__":
	# # 添加
	# path = '../../resources/dataset/local/chitchat/retrieval/merge_all/opensource_merge_v3.txt'
	# lines = get_data(path)
	# recaller.insert_bulk('chitchat-v1', lines)
	
	# 查询
	res = recaller.search_by_match(query='我好伤心啊', size=3, index='chitchat-v1', match_item='question')
	print(res)
	
	# 删除
	# #es_delete(ES_TEST_INDEX)
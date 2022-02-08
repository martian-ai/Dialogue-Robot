
import sys, os
sys.path.append('../..')

from modules.basic_search.recall.function.recaller import EsRecaller
from modules.structure_document.document import get_data

recaller = EsRecaller(ip='10.9.227.9', port='8200')

if __name__ == "__main__":

    line = '我特别厉害'
    res = recaller.search_by_match(query=line, size=200, index='chitchat-v1', match_item='question')

    for item in res:
        print(item["_source"]["answer"].strip().replace("。",""))

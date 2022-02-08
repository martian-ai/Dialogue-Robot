
import sys, os
sys.path.append('../..')

from modules.basic_search.recall.function.recaller import EsRecaller
from modules.structure_document.document import get_data

recaller = EsRecaller(ip='10.9.227.9', port='8200')

if __name__ == "__main__":

    base_dir = "../../modules/alpha_topic/bertopic/场景输出/场景-相似问-1.2"
    file_list = os.listdir(base_dir)

    for item in file_list:
        print(item)
        domain_res = []
        with open(os.path.join(base_dir,item), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                res = recaller.search_by_match(query=line, size=10, index='chitchat-v1', match_item='question')
                # print(res[0]["_score"])
                if res[0]["_score"] > 10:
                    domain_res.append(res[0]["_source"]["answer"].strip().replace("。",""))
                if res[1]["_score"] > 10:
                    domain_res.append(res[1]["_source"]["answer"].strip().replace("。", ""))
                if res[2]["_score"] > 10:
                    domain_res.append(res[2]["_source"]["answer"].strip().replace("。",""))

        domain_res = sorted(set(domain_res))

        with open( "res_" + item, mode='w', encoding='utf-8') as f:
            for item in domain_res:
                if len(item) < 6:
                    continue
                if len(item) > 30:
                    continue
                if "'" in item:
                    continue
                if '"' in item :
                    continue
                if ':' in item :
                    continue
                if '?' in item :
                    continue
                if "置顶" in item:
                    continue
                f.write(item + '\n')
	# 删除
	# #es_delete(ES_TEST_INDEX)
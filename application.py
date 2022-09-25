# -*- coding: UTF-8 -*-

import json
from pathlib import Path

import config
from flask import Flask, request
from flask_cors import CORS, cross_origin

from modules.structure_document.base_doc import BaseDoc
from solutions.service_faq.FAQRecall import FAQRecall

app = Flask(__name__)
app.config.from_object(config)
CORS(app)

# faq_recaller = FAQRecall(ip='127.0.0.1', port='9200')
#faq_recaller = FAQRecall(ip='10.9.227.9', port='8200')


@app.route('/', methods=['POST', 'GET'])
def root():
    return 'welcome to bot application'


"""
Part I : qa mining interface
"""


@app.route('/doc_qa_mining', methods=['GET', 'POST'])
def interface_doc_qa_mining():
    inputDoc = request.args.get("inputDoc")
    topK = request.args.get("topK")
    print(inputDoc)
    baseDoc = BaseDoc(inputDoc)
    # baseDoc.get_sum()
    baseDoc.doc_qa_mining(topK)
    return json.dumps({'inputDoc': baseDoc.text, 'summary': baseDoc.summary,  'results': baseDoc.faqs}, ensure_ascii=False)

# {"inputDoc": "百度二字，来自于八百年前南宋词人辛弃疾的一句词：众里寻他千百度。这句话描述了词人对理想的执着追求。1999年底，身在美国硅谷的李彦宏看到了中国互联网及中文搜索引擎服务的巨大发展潜力，抱着技术改变世界的梦想，他毅然辞掉硅谷的高薪工作，携搜索引擎专利技术，于 2000年1月1日在中关村创建了百度公司。", "summary": "1999年底，身在美国硅谷的李彦宏看到了中国互联网及中文搜索引擎服务的巨大发展潜力，抱着技术改变世界的梦想，他毅然辞掉硅谷的高薪工作，携搜索引擎专利技术，于 2000年1月1日在中关村创建了百度公司百度二字，来自于八百年前南宋词人辛弃疾的一句词：众里寻他千百度", "results": [["百度是怎么创建的", "1999年底，身在美国硅谷的李彦宏看到了中国互联网及中文搜索引擎服务的巨大发展潜力，抱着技术改变世界的梦想，他毅然辞掉硅谷的高薪工作，携搜索引擎专利技术，于 2000年1月1日在中关村创建了百度公司"], ["百度二字来自于什么？", "百度二字，来自于八百年前南宋词人辛弃疾的一句词：众里寻他千百度"]]}

#TODO doc kb mining

"""
Part II FAQ
"""
# @app.route('/faq', methods=['GET', 'POST'])
# def interface_faq():
#     query = request.args.get("query")
#     score_list, question_list, answer_list = faq_recaller.search_by_qa(query, 'doc_instruction', "question", 1)
#     # score_list, question_list, answer_list = faqRecall(query)
#     question = question_list[0]
#     answer = answer_list[0]
#     return json.dumps({'query': query, 'question': question, 'answer': answer}, ensure_ascii=False)


"""
Part IV doc qa
"""


"""
Part V orqa
"""

"""
Part III task bot
"""

# def chitchat(conv, up, query):
#     query = request.args.get("inputQuery")
#     results = es_search(query, app.config['ES_INDEX_DOC_ThreeBody']) #TODO
#     return results

# def dialog():
#     conv = Conversation()
#     up = UserProfile()

#     while True:
#         query = input("Please Enter:")
#         # query_rewrite = query_rewrite(query)
#         query_intent = query_intent(conv, up, query) # TODO Ranking or CLF
#         while not query:
#             print('Input should not be empty!')
#             query = input("Please Enter:")
#         (a, b), c = predict(query, qa_model, 'bert-base-chinese', qa_device, qa_tokenizer, 3, 16, 24, 384, 3, 128, False, 'document-threebody', True)

#         conv.update_conv(summary)
#         up.update_up(keywords)


if __name__ == '__main__':
    """
    test 1 qa
    """

    """
    test 2 qg
    """
    # query = '云天明送给程心的行星叫什么？'
    # query = '范廷颂于1919年6月15日在越南宁平省天主教发艳教区出生；童年时接受良好教育后，被一位越南神父带到河内继续其学业\n范廷颂于1940年在河内大修道院完成神学学业'
    # query = '她最后发现自己来到了阮雯的家门前，在大学四年中，阮老师一直是她的班主任，也是她最亲密的朋友\n在叶文洁读天体物理专业研究生的两年里，再到后来停课闹革命至今，阮老师一直是她除父亲外最亲近的人\n阮雯曾留学剑桥，她的家曾对叶文洁充满了吸引力，那里有许多从欧洲带回来的精致的书籍、油画和唱片，一架钢琴；还有一排放在精致小木架上的欧式烟斗，父亲那只就是她送的，这些烟斗有地中海石楠根的，有土耳其海泡石的，每一个都仿佛浸透了曾将它们拿在手中和含在嘴里深思的那个男人的智慧，但阮雯从未提起过他。这个雅致温暖的小世界成为文洁逃避尘世风暴的港湾\n但那是阮雯的家被抄之前的事，她在运动中受到的冲击和文洁父亲一样重，在批斗会上，红卫兵把高跟鞋挂到她脖子上，用口红在她的脸上划出许多道子，以展示她那腐朽的资产阶级生活方式。'
    # query = '文件的提供者是叶文洁的妹妹叶文雪\n作为一名最激进的红卫兵，叶文雪积极主动地揭发父亲，写过大量的检举材料，其中的一些直接导致了父亲的惨死\n但这一份材料文洁一眼就看出不是妹妹写的，文雪揭发父亲的材料文笔激烈，读那一行行字就像听着一挂挂炸响的鞭炮，但这份材料写得很冷静、很老到，内容翔实精确，谁谁谁哪年哪月哪日在哪里见了谁谁谁又谈了什么，外行人看去像一本平淡的流水账，但其中暗藏的杀机，绝非叶文雪那套小孩子把戏所能相比的。'
    # query = '叶文洁的妹妹是叶文雪\n叶文洁的妹妹是叶文雪'

    # print(es_search('云天明送给程心的行星叫什么？', app.config['ES_INDEX_DOC_ThreeBody']))
    # cmd_line_interactive()

    app.run('0.0.0.0', 5061, debug=True)

'''
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2023 by Martain.AI, All Rights Reserved.
#
Description: # 
Author: # apollo2mars apollo2mars@gmail.com
################################################################################
'''

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec2": "uer/sbert-base-chinese-nli",
    "text2vec3": "../../resources/model/text2vec3",
}

class LangchainArgument(object):
    web_sina_finance_http = 'https://finance.sina.com.cn/'
    web_baidu_http = "https://baidu.com.cn/"
'''
Description: 
Version: 
Author: Apollo
Date: 2021-12-15 18:46:29
LastEditors: sueRim
LastEditTime: 2021-12-15 18:54:10
Todo: 
'''

# -*- coding: utf-8 -*-
import requests
import json
import urllib
import http.client
import hashlib
import random
import time

def baidu_translate(content, fromLang, toLang):
    if not content:
        return ''
    appid = '20190313000276689'  # 填写你的appid
    secretKey = 'v0COhfoWBpHsRqoESwLy'  # 填写你的密钥
    httpClient = None
    myurl = '/api/trans/vip/translate'
    salt = random.randint(32768, 65536)
    sign = appid + content + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(content) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign
    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)
        dst = str(result["trans_result"][0]["dst"])  # 取得翻译后的文本结果
        return dst
    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()

# 回译
def deal_translate(line):
    expandSentenceList = []
    if line:
        expandSentence = baidu_translate(line, "zh", "en")
        print(expandSentence)
        time.sleep(1)
        if expandSentence:
            expandSentence = baidu_translate(expandSentence, "en", "zh")
            time.sleep(4)
            expandSentenceList.append(expandSentence)
    return expandSentenceList

if __name__ == "__main__":
    deal_translate('你好')
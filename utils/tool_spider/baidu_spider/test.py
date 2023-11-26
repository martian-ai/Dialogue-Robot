'''
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2023 by Martain.AI, All Rights Reserved.
#
Description: 
Author: apollo2mars apollo2mars@gmail.com
################################################################################
'''
# 导入BaiduSpider
from baiduspider import BaiduSpider
from pprint import pprint

# 实例化BaiduSpider
spider = BaiduSpider()

# 搜索网页
pprint(spider.search_web(input('input:')).plain)

# pprint(BaiduSpider().search_zhidao(input('搜索词：')).plain)
# pprint(BaiduSpider().search_zhidao(input('搜索词：')).plain)

# pprint(spider.sear)
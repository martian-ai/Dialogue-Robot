#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved.

mini spider main module

Authors: sunhongchao(sunhongchao@baidu.com)
Date: 2022/11/22 17:23:06
"""

import re
from urllib.parse import urlparse
from urllib.parse import urljoin
from logger import logger
from page_retriever import PageRetriever


class PageParser(object):
    """解析网页.

    输出有效的子链接
    输入一个url，PageParser将会调用PageRetriever下载该url文档，然后打开本地的磁盘文档，
    将所有href索引链接解析出来，去除无效链接后返回

    Attributes:
        html_url: 要解析的url
        pattern: 正则匹配模式， 默认解析的链接为r'''href=["']?[^\s<>]+'''
        即从文档中提取所遇的href索引
    """

    def __init__(self, html_url, pattern=r'''href=["']?[^\s<>]+'''):
        """初始化."""
        self._pattern = pattern
        self._url = html_url

    def url_read(self):
        """从文件中读取url.

        将下载到本地的文档读到内存
        返回一串字符

        Returns:
            string: url字符串
        """
        logger.info("Parse %s" % self._url)
        retriever = PageRetriever(self._url)
        res = retriever.download()  # 本地存储
        file_content = ""

        if res is None:
            return file_content

        (file_name, headers) = res

        try:
            if headers.get_content_type() == 'text/html':
                with open(file_name, 'rb') as f:
                    file_content = f.read()
        except IOError as e:
            logger.error("Open file error :\n %s" % e)
        except UnicodeDecodeError:
            logger.error("Decode failed! ")

        return str(file_content)

    def parse(self):
        """url解析.

        解析文档中的href链接，返回符合规则的链接

        Returns:
            set: 满足规则的链接
        """

        content = self.url_read()
        link_pattern = r'''href=["']?[^\s<>]+'''
        urls = re.findall(link_pattern, content)

        links = set()
        for url in urls:
            relative_parts = url[5:]  # remove href=''
            if "href" in relative_parts:
                pos = relative_parts.index("href")
                relative_parts = relative_parts[pos + len("href="):]
                # handle javascript:location.href
            relative_parts = relative_parts.strip('\\\'\"')  # remove ' or "
            standard_link = urljoin(self._url, relative_parts)
            (scheme, netloc, path, params, query, fragment) = urlparse(url)
            if scheme not in ('', 'https', 'http'):
                logger.info("%s is not supported. Ignored" % url)
                continue

            links.add(standard_link)
        return links

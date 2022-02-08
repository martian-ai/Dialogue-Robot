#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved.

mini spider main module

Authors: sunhongchao(sunhongchao@baidu.com)
Date: 2022/11/22 17:23:06
"""

from logger import logger
from config_reader import ConfReader

from urllib.request import urlretrieve
from urllib.request import urlcleanup
from urllib.request import splitport

from urllib.parse import urlparse
from urllib.error import URLError
from urllib.error import HTTPError

from os.path import splitext
from os.path import dirname
from os.path import sep
from os.path import isdir
from os.path import exists
from os.path import join
from os import removedirs
from os import makedirs
from os import unlink

from re import compile
from tld import get_tld
from tld import exceptions


class PageRetriever(object):
    """页面下载并保存.

    Attributes:
        _url (string): 要下载的url
        _root_path: 存储的路径
    """

    def __init__(self, url):
        """初始化函数.

        Args:
            url (_type_): _description_
        """
        self._url = str(url).strip('\n')
        self._root_path = ConfReader.instance().get_output_directory()

    def download(self):
        """下载指定的文件格式的文档到本地路径.

        Returns:
            (file, header): 下载指定的文件格式的文档到本地路径
        """
        file_name = self.save_filename()
        if file_name is None:
            return

        try:
            target_url_pattern = ConfReader.instance().get_target_urls()
            pattern = compile(target_url_pattern)
            result = pattern.search(self._url)
            if result is None:
                # logger.info("Supported pattern is %s, but the url is %r" %
                #            (target_url_pattern, self.__url))
                res = urlretrieve(self._url)
            else:
                res = urlretrieve(self._url, file_name)  # download page
        except (IOError, ValueError, HTTPError, URLError) as e:
            logger.warning("Url retrieve failed %s. %s. Removed invalid dir" %
                           (self._url, e))
            if exists(file_name):
                unlink(file_name)
            else:
                try:
                    invalid_dir = dirname(file_name)
                    if exists(invalid_dir):
                        removedirs(invalid_dir)
                except OSError as e:
                    logger.error(e)
                    pass

            return None
        return res

    def save_filename(self, index_file='index.html'):
        """页面本地存储.

        存储方式：
           按照域名的路径建立本地树形目录，域名为根目录，分类名为一级目录
           如果空，则分类名使用域名。
           example: http://www.wikipedia.org/bike/chendonghua
           本地存储的目录结构为：
               wikipedia.org
               --bike
               ----chendonghua
           又如： https:www.somenet.com/somepage.html
           本地存储的目录结构为：
               somenet.com
               --somenet.com
               ----somepage.html

        Args:
            index_file (str, optional): _description_. Defaults to 'index.html'

        Returns:
            str: 页面本地文件路径
        """
        if self._url is None:
            return None

        logger.info("Save url %s " % self._url)
        (scheme, netloc, path, params, query, fragment) = urlparse(self._url)

        try:
            host, port = splitport(netloc)
            hostname = get_tld(self._url, fail_silently=True)
            #当域名是Ip形式表示的时候， 会抛出异常， fail_silently参数抑制这个异常
            #此时 函数返回一个None
            if hostname is None:
                category = hostname = netloc
            else:
                category = host.replace(hostname, '').strip('.')
                if len(category) == 0:
                    category = host
        except exceptions.TldBadUrl as e:
            logger.error("Url is bad : %s" % self._url)
            return None

        if len(path) == 0:
            path = "/"  # default character

        local_path = join(hostname, category + path)
        (root, ext) = splitext(local_path)
        if ext == '':  # there is no file
            if local_path[-1] == '/':
                #logger.info("Url %s is not a file" % self.__url)
                local_path += index_file
            else:
                local_path += '/' + index_file

        local_dir = dirname(join(self._root_path, local_path))
        if sep != '/':
            local_dir.replace("/", sep)

        try:
            if not isdir(local_dir):
                if exists(local_dir):
                    unlink(local_dir)
                makedirs(local_dir)
        except IOError as e:
            logger.error(e)
            print("Make dir {} failed".format(local_dir), e)
        return join(self._root_path, local_path)

    def clear(self):
        """清除本地临时文件."""
        urlcleanup()
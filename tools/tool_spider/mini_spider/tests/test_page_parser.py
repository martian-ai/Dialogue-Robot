#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved.

mini spider main module

Authors: sunhongchao(sunhongchao@baidu.com)
Date: 2022/11/22 17:23:06
"""

import unittest
import shutil
import os

from config_reader import ConfReader
from page_parser import PageParser

class TestPageParser(unittest.TestCase):
    """PageParser测试类."""

    def tearDown(self):
        """文件清除."""
        output_dir = ConfReader.instance().get_output_directory()
        if os.path.exists(output_dir):
            shutil.rmtree(ConfReader.instance().get_output_directory())

    def test_parse(self):
        """文件解析测试."""
        url1 = 'localhost:8081/page1.html'
        expect_sub_url = 'localhost:8081/1/page1_1.html'
        parser = PageParser(url1)
        links = parser.parse()
        self.assertIn(expect_sub_url, links)

        url2 = 'localhost:8081/page7.html'
        parser = PageParser(url2)
        links = parser.parse()
        self.assertEqual(links, set())

        url3 = 'localhost:8081/3/image.jpg'
        parser = PageParser(url3)
        self.assertEqual(parser.parse(), set())

    def test_url_read(self):
        """测试url读取.

        测试了三个场景：
        使用标准url
        使用无效url
        使用其他格式的url文档，如jpg

        """
        url1 = 'localhost:8081/page1.html'
        parser = PageParser(url1)
        content1 = parser.url_read()
        self.assertEqual(content1.__contains__('page1_4.html'), True)
        self.assertEqual(content1.__contains__('page1_1.html'), True)

        #invalid url test
        url2 = 'localhost:8081/page7.html'
        parser = PageParser(url2)
        content2 = parser.url_read()
        self.assertEqual(content2, '', "return content should be empty")

        #No support url test
        url3 = 'localhost:8081/3/image.jpg'
        parser = PageParser(url3)
        content3 = parser.url_read()
        self.assertEqual(content3, '')
        self.assertLogs(logger='../logs/spider.log', level='error')

if __name__ == '__main__':
    unittest.main()

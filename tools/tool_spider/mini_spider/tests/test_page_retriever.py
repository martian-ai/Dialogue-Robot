#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved.

mini spider main module

Authors: sunhongchao(sunhongchao@baidu.com)
Date: 2022/11/22 17:23:06
"""

import unittest
import os
import shutil

from page_retriever import PageRetriever
from config_reader import ConfReader


class TestPageRetriever(unittest.TestCase):
    """PageRetriever测试类."""

    def setUp(self):
        """"""
        pass

    def tearDown(self):
        """文件清除."""
        output_dir = ConfReader.instance().get_output_directory()
        if os.path.exists(output_dir):
            shutil.rmtree(ConfReader.instance().get_output_directory())

    def test_download(self):
        """测试url下载."""
        r1 = PageRetriever('localhost:8081/page1.html')
        filename, headers = r1.download()

        self.assertEqual(os.path.exists(filename), True)
        self.assertEqual(headers.get_content_type(), 'text/html')

        invalid_url = 'http://invalid_url/invlid/idfn/a.html'
        r2 = PageRetriever(invalid_url)
        res = r2.download()
        self.assertIsNone(res)
        local_path = ConfReader.instance().get_output_directory()
        maybe_dir = os.path.join(local_path, 'invalid_url')
        self.assertEqual(os.path.exists(maybe_dir), False)

        r3 = PageRetriever('localhost:8081/')
        res = r3.download()
        self.assertIsNotNone(res)
        self.assertEqual(res[1].get_content_type(), 'text/html')


if __name__ == '__main__':
    unittest.main()

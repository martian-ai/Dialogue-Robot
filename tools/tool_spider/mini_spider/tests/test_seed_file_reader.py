#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved.

mini spider main module

Authors: sunhongchao(sunhongchao@baidu.com)
Date: 2022/11/22 17:23:06
"""

from seedfile_reader import SeedFileReader
import unittest


class TestSeedFileReader(unittest.TestCase):
    """SeedFileReader测试类."""
    
    def test_read(self):
        """测试种子文件读取."""
        reader = SeedFileReader('seeds.txt')
        g = reader.read()
        for line in g:
            self.assertIsInstance(line, str)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

mini spider main module

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

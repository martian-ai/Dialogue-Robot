#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

mini spider main module

"""

import os
from config_reader import ConfReader


class SeedFileReader(object):
    """种子文件读取类.

    Args:
        _file_name (str): 种子文件路径
    """

    def __init__(self, file_name='seeds.txt'):
        cur_file = os.path.abspath(__file__)
        cur_dir = os.path.join(os.path.dirname(cur_file),
                               ConfReader.instance().get_url_list_file())
        self._file_name = os.path.join(cur_dir, file_name)  # absolute path

    def read(self):
        """返回文件中的一行.

        Yields:
            str: 文件中的一行
        """
        with open(self._file_name, 'r') as f:
            for each_line in f:
                yield each_line


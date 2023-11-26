#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

mini spider main module

"""

import optparse
import time

from crawler_thread import CrawlerThreadPool
from seedfile_reader import SeedFileReader

VERSION = "1.0"


def main():
    """ mini spider main 方法，方便其他函数调用"""
    start_time = time.time()
    parser = optparse.OptionParser(usage="%prog [-f] [-q]",
                                   version="%prog-{}".format(VERSION))  # 参数解析
    parser.add_option("-c",
                      dest="conf",
                      type="string",
                      help="Specify a conf file")
    (options, args) = parser.parse_args()
    seed_file = SeedFileReader()  # 读取种子文件
    if not options.conf:
        print("You have no specified any conf file. Using default conf file")
    pool = CrawlerThreadPool(config_name=options.conf)  # 创建线程池
    g = seed_file.read()
    for seed in g:
        seed = seed.strip('\n')
        pool.add_seed(seed)
    pool.begin()
    end_time = time.time()
    print("Spend time {} seconds".format(round(end_time - start_time, 2)))


if __name__ == '__main__':
    main()

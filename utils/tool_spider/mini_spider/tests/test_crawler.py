#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mini spider main module

"""

import unittest
import shutil
import os
import time
from queue import Queue

from crawler_thread import CrawlerThread
from crawler_thread import CrawlerThreadPool
from config_reader import ConfReader


class TestCrawlerThread(unittest.TestCase):
    """CrawlThread测试类."""

    def setUp(self):
        """测试前的准备工作."""

    def tearDown(self):
        """文件清除."""
        output_dir = ConfReader.instance().get_output_directory()
        if os.path.exists(output_dir):
            shutil.rmtree(ConfReader.instance().get_output_directory())

    def test_crawl(self):
        """爬取测试."""
        url1 = "http://localhost:8081"
        q = Queue()
        q.put(url1)
        thread1 = CrawlerThread(q)
        thread1.crawl((url1, 0))
        self.assertEqual(CrawlerThreadPool.total_links, 5)

        # depth exceed test
        url2 = 'localhost:8081/mirror/page1.html'
        thread2 = CrawlerThread(q)
        thread2.crawl((url2, 101))
        self.assertEqual(CrawlerThreadPool.total_links, 5)

        url3 = 'http://www.baidu.com?query=10000'
        thread3 = CrawlerThread(q)
        CrawlerThreadPool.interval_links_cnt = \
            ConfReader.instance().get_max_links_count() + 1
        t1 = time.time()
        thread3.crawl((url3, 0))
        t2 = time.time()
        self.assertAlmostEqual(t2 - t1,
                               ConfReader.instance().get_crawl_interval(), 0)


if __name__ == '__main__':
    unittest.main()

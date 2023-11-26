# -*- coding: utf-8 -*-
"""

config reader module

"""

import configparser
import os
from logger import logger


class ConfReader(object):
    """配置文件的读取类.

    Args:
        object (_type_): 继承自object

    Returns:
        _type_: _description_
    """
    @classmethod
    def instance(cls, config_name='spider.cfg'):
        """配置文件读取.

        使用单例方式，只读一次，放在内存中.

        Args:
            config_name (str, optional): 配置文件名，无需指定路径，只要文件名即可, 默认是'spider.cfg'
        Returns:
            _type_: _description_
        """

        if not hasattr(cls, "_instance"):
            cls._instance = cls(config_name)
        return cls._instance

    def __init__(self, config_name):
        """配置文件读取类.

        Args:
            config_name (string): 配置文件名
        """
        dir_name = os.path.abspath(__file__)
        self._conf_dir = os.path.join(os.path.dirname(dir_name), 'conf')
        self._url_list_file = "./urls"
        self._output_directory = "./output"
        self._max_depth = 1
        self._crawl_interval = 1
        self._crawl_timeout = 1
        self._target_urls = ".*.(htm|html)"
        self._thread_count = 4
        self._max_links_count = 10000
        self.__read_confs(config_name)

    def __read_confs(self, file_name):
        """通过ConfigParser读取配置文件.

        Args:
            file_name (str): 读取的文件名字
        """
        config = configparser.ConfigParser()
        config.read(os.path.join(self._conf_dir, file_name), encoding='utf-8')

        section_name = 'spider'
        if section_name not in config.sections():
            logger.warning("Read conf failed. Used default values")
            return
        else:
            self._url_list_file = config.get(section_name, "url_list_file")
            self._output_directory = config.get(section_name,
                                                "output_directory")
            self._max_depth = config.getint(section_name, "max_depth")
            self._crawl_interval = config.getint(section_name,
                                                 "crawl_interval")
            self._target_urls = config.get(section_name, "target_url")
            self._thread_count = config.getint(section_name, "thread_count")
            self._max_links_count = config.getint(section_name, "max_link")

    def get_url_list_file(self):
        """返回种子节点所在的文件夹.

        Returns:
            string : 种子节点所在的文件夹
        """
        return self._url_list_file

    def get_output_directory(self):
        """返回输出数据文件夹的路径.

        Returns:
            string : 输出数据文件夹的路径
        """
        return self._output_directory

    def get_max_depth(self):
        """返回最大抓取深度.

        Returns:
            int: 最大抓取深度
        """
        return self._max_depth

    def get_crawl_interval(self):
        """返回线程抓取间隔.

        间隔时间内抓取量超过一定数量时，线程沉睡一定的间隔时间， 单位 秒

        Returns:
            int : 线程抓取间隔
        """
        return self._crawl_interval

    def get_target_urls(self):
        """返回目标url的正则匹配模式.

        Returns:
            int: 目标url的正则匹配模式 
        """
        return self._target_urls

    def get_thread_count(self):
        """返回开启线程的数量.

        Returns:
            int: 开启线程的数量
        """
        return self._thread_count

    def get_max_links_count(self):
        """返回每次间隔时间内的链接抓取量.

        Returns:
            int: 每次间隔时间内的链接抓取量
        """
        return self._max_links_count


if __name__ == '__main__':

    ths = ConfReader.instance().get_thread_count()
    print(ths)

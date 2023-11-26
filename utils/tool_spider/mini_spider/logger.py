# -*- coding: utf-8 -*-
"""

logger module

"""

import logging.handlers
import os
from os.path import dirname
from os.path import abspath
from os.path import join

VERSION = "1.0.0"


# 以文件的方式存储日志
LOG_FILE = join(dirname(abspath(__file__)), 'logs' + os.sep + 'spider.log')
logger = logging.getLogger(LOG_FILE)

# 设置日志记录级别
logger.setLevel(logging.DEBUG)

# 设置日志存储格式
fmt = '[%(asctime)s]-[%(levelname)s]-[%(processName)s:%(process)s] %(message)s'
formatter = logging.Formatter(fmt)

# RotatingFileHandler 将日志消息发送到磁盘文件，并支持日志文件按大小切割，大小 100M
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=100 * 1024 * 1024, backupCount=2)
handler.setFormatter(formatter)
logger.addHandler(handler)
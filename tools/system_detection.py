#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019-01-29 16:24
# @Author  : apollo2mars
# @File    : system_detection.py
# @Contact : apollo2mars@gmail.com
# @Desc    : detect system type

import platform

sysstr = platform.system()

if sysstr is 'Darwin':
    print("use mac os")
elif sysstr is "Linux":
    print("use Linux os")
else:
    print(sysstr)
#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019-03-12 18:40
# @Author  : apollo2mars
# @File    : data_v.py
# @Contact : apollo2mars@gmail.com
# @Desc    : data visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_excel_and_draw_pie(filename):

	df = pd.read_excel(filename, encoding='utf-8')
	# print(df.head())
	a = df.groupby('Domain').size()
	# print('statistic of domain data', df.groupby('Domain').size())
	# print('statistic of intent data', df.groupby(['Domain','Intent']).size())
	'''
	visualization
	'''
	a.plot.pie(figsize=(10, 10), autopct='%.2f')
	# http://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
	plt.show()


def read_list_and_draw_pie(input_list:list):
	name_list = []
	num_list = []
	for item in input_list:
		name_list.append(item[0])
		num_list.append(item[1])
	print(num_list)
	print(name_list)
	# 保证圆形
	plt.axes(aspect=1)
	plt.pie(x=num_list, labels=name_list, autopct='%3.1f %%')
	plt.show()
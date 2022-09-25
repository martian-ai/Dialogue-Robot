"""
闲聊语料处理
"""
def get_data(filename):
	lines = []
	with open(filename, encoding='utf-8', mode='r') as f:
		lines = f.readlines()
	return lines

def line_chunking(lines):
	"""
	当前仅实现简单版本
	TODO 复杂版本调研，chunking 模块和后续模块联合调用
	TODO 返回结果使用generator
	"""

	all_results = []
	for line in lines:
		line = line.strip()
		if len(line) == 0:
			continue
		all_results.append(line)
	return all_results

if __name__ == "__main__":
	lines = get_data('../../resources/corpus/chitchat/format_v3.txt')
	print(len(lines))
	paras = line_chunking(lines)
	print(len(paras))
	print(paras[0])
	print(paras[-1])
		
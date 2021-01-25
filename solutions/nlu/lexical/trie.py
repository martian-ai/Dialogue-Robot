class Node(object):
    """
    TriedTree节点
    """
    def __init__(self, name):
        """
        前缀树节点
        :param name: 节点名，除了根节点是root 其他都是字符
        """
        self.name = name
        self.children = {}
        self.is_word = False
        self.tag = ""

class TriedTree(object):
    """
    TriedTree
    """
    def __init__(self):
        self.root = Node("root")

    def insert(self, word, tag):
        """
        插入单词
        :param word:单词
        :return:
        """

        if word == "":
            return
        word = list(word)

        def do_insert(i, node):
            """
            递归插入单词
            :param i: 位置索引
            :param node:字母节点
            :return:
            """
            if i == len(word):
                node.is_word = True
                node.tag = tag
                return
            sub_node = node.children.get(word[i])
            if sub_node is None:
                sub_node = Node(word[i])
                node.children[word[i]] = sub_node
            do_insert(i + 1, sub_node)
            # char_val = word_list[0]

        index = 0
        first_node = self.root.children.get(word[index])
        if first_node is None:
            first_node = Node(word[index])
            self.root.children[word[index]] = first_node
        do_insert(index + 1, first_node)

    def segment_word(self, sentence):
        """
        检测关键词，分词并且打标
        :param sentence:
        :return:
        """
        index = 0
        result = []

        if sentence == "":
            return result
        sentence = list(sentence)

        def deep_first_search(i, node):
            """
            深度优先搜索目标节点返回下标和节点标签
            :param i:
            :param node:
            :return:
            """
            # if node.children is None:
            #     return i
            if i == len(sentence) and node.is_word:
                return i, True, node.tag
            if i == len(sentence) and not node.is_word:
                return i, False, ""
            sub_node = node.children.get(sentence[i])
            if sub_node is None and node.is_word:
                return i, True, node.tag
            if sub_node is None and not node.is_word:
                return i, False, ""
            return deep_first_search(i + 1, sub_node)

        while index < len(sentence):
            first_node = self.root.children.get(sentence[index])
            begin = index
            index += 1
            if first_node is None:
                continue
            end, success, tag = deep_first_search(index, first_node)
            index = end
            if not success:
                continue
            result.append({"word": "".join(sentence[begin:end]), "tag": tag, "begin": begin, "end": end})

        return result

def get_pre_intervals(pre_seg):
    pre_intervals = []
    pre_index = 0
    for item in pre_seg:
        pre_intervals.append([pre_index, pre_index + len(item)])
        pre_index += len(item)
    return pre_intervals

def check_intervals_contain(single, intervals):
    for item in intervals:
        if int(item[0]) <= int(single[0]) and int(single[1]) <= int(single[1]):
            return True
    return False

def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][-1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][-1] = max(merged[-1][-1], interval[-1])
    return merged

def merge_sub(intervals_sub, intervals_all):
    tmp = []
    for item2 in intervals_all:
        for item1 in intervals_sub:
            if item1[0] < item2[0] < item1[1] or item1[0] < item2[1] < item1[1]:
                tmp.append(item2)
                continue
    return tmp

def find_merge_intervals(new_intervals, pre_intervals):
    sub_merge_intervals = merge(merge_sub(new_intervals, pre_intervals))
    tmp = merge_sub(new_intervals, pre_intervals)
    merge_intervals = []
    for item in pre_intervals:
        if item in tmp:
            continue
        else:
            merge_intervals.append(item)
    return merge_intervals, sub_merge_intervals

def get_results(input_text, pre_seg, tree):
    # pre_intervals [[0, 1], [1, 2], [2, 5], [5, 9], [9, 10], [10, 11], [11, 12], [12, 14], [14, 16], [16, 17], [17, 19], [19, 22], [22, 23]]
    # intervals [[2,9], [14, 16]]
    # 合并逻辑
    #  1. 如果 intervals 为空，则返回pre_seg 的结果 
    #  2. 如果 intervals 中的某一个元素 在 pre_intervals 中，则将 intervals 中的词元素删除。
    #       例子 intervals 中的 [14, 16] 在 pre_intervals 中， 需要将 intervals 中的 [14,16]删除
    #  3. 如果 intervals 中的某一个元素 被 包含于 pre_intervals 的某个元素所在区间中，则不进行任何处理，不强制进行切分， 需要将intervals 中的该元素删除。  当前未使用
    #  4. 如果 intervals 中的某个元素 包含 pre_intervals 中的若干个元素所对应的区间，或者有交叉，则对区间进行再次合并，使用trie 树合并后的区间中的文本再次进行分词

    intervals = [[item['begin'], item['end']] for item in tree.segment_word(input_text)]
    #print("与 trie 树匹配的区间: ",intervals)

    if not intervals: # 1. intervals 为空
        return pre_seg

    pre_intervals = get_pre_intervals(pre_seg) 
    #print("模型分词后的区间: ", pre_intervals)

    new_intervals = []
    for item in intervals:
        if item in pre_intervals: # 2. 如果 intervals 中的某一个元素 在 pre_intervals 中
            continue
        # elif check_intervals_contain(item, pre_intervals): # 3. 如果 intervals 中的某一个元素 被 包含于 pre_intervals 的某个元素所在区间中
        #     print("BBB ", item)
        #     continue
        else:
            new_intervals.append(item)
    #print("去除 trie 与模型 相同的区间: ", new_intervals)

    merge_intervals, sub_merge_intervals = find_merge_intervals(new_intervals, pre_intervals)
    #print("有待用trie 结果替换的区间:", sub_merge_intervals)
    #print("合并后的区间:", merge_intervals)
    results = []
    for item in merge_intervals:
        item_text = input_text[item[0]:item[1]]
        if item in sub_merge_intervals:
            cut_index = []
            cut_index.append(0) # 1
            cut_index.append(item[1]-item[0]) # 6
            for tmp in tree.segment_word(item_text):
                cut_index.append(tmp['begin'])
                cut_index.append(tmp['end'])
            cut_index = sorted(list(set(cut_index)))
            for idx in range(0, len(cut_index)-1):
                __tmp = item_text[cut_index[idx]:cut_index[idx+1]]
                results.append(input_text[cut_index[idx]:cut_index[idx+1]])
        else:
            tmp_text = input_text[item[0]:item[1]]
            results.append(tmp_text)
    return results

def load_tree(path):
    with open(path, mode='r', encoding='utf-8') as f:
        s = [  item.strip() for item in f.readlines()]
    tree = TriedTree()
    for item in s:
        tree.insert(item, "word")
    return tree


if __name__ == '__main__':

    with open('zlzk-domain-dict-1210.txt', mode='r', encoding='utf-8') as f:
        s = [  item.strip() for item in f.readlines()]
    tree = TriedTree()
    for item in s:
        tree.insert(item, "word")
    # tree.insert("a", "char")
    #input_text = "我是南京市长江大桥，我是一个市长，工作二十年了"
    # input_text = "检查雨刮开关有电到开关"
    input_text ='副梁开裂，计划已到，客户修理厂更换。'	
    # ground = ['副梁', '开裂', '，', '计划', '已到', '，', '客户', '修理厂', '更换', '。']
    pred = ['副梁', '开裂', '，', '计划', '已', '到', '，', '客户', '修理', '厂', '更换', '。']	
    #ground = ['中回', '接头', '油封', '损坏', '导致', '不','正常']
    #ground = ['客户', '修理厂', '更换', '。']
    #['中', '回接', '头油', '封损', '坏导', '致', '不正']

    results = get_results(input_text, pred, tree)
    print(results)

    # import jieba
    # print(list(jieba.cut(input_text)))

"""
jaccard word and char literal similarity 
"""

def common_char(text1, text2):
    set1 = set(list(text1))
    set2 = set(list(text2))
    intersection = set1.intersection(set2)
    return intersection

def common_word(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    return intersection

def char(text1, text2):
    intersection = common_char(text1, text2)
    return len(intersection)/(len(text1) + len(text2) - len(intersection))

def word(llist1, list2):
    intersection = common_char(llist1, list2)
    return len(intersection)/(len(llist1) + len(list2) - len(intersection))


if __name__ == '__main__':
    print(char('123', '111'))
    print(word(['1', '2', '3'], ['1', '1', '1']))
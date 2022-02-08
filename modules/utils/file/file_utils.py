import os

def get_dir_file_list(local_dir, suf=None):
    """
            获取文件夹下的全部文件
            :param local_dir: 本地文件夹路径
            :param suf: 提取固定后缀的文件路径，默认不考虑后缀
            :return:
            """
    if not os.path.exists(local_dir):
        raise Exception('获取文件夹下的文件失败，文件夹不存在！ local_dir:[%s]' % local_dir)

    result_path_list = []
    for file in os.listdir(local_dir):
        real_path = '/'.join((local_dir, file))
        if os.path.isdir(real_path):
            res_path_list = get_dir_file_list(real_path, suf)
            for res_item_path in res_path_list:
                result_path_list.append(res_item_path)
        else:
            if suf is None:
                result_path_list.append(real_path)
            elif real_path.endswith(suf):
                result_path_list.append(real_path)
    return result_path_list

def merge_same_suf_text_file(file_dir, end_file_path, suf):
    """
    合并同后缀的文本文件
            :param file_dir: 文件目录
            :param end_file_path: 合并后的文件名
            :param suf: 文件后缀
            :return: 无
            """
    file_path_list = get_dir_file_list(file_dir, suf)
    end_file = open(end_file_path, 'w')
    for item_file_path in file_path_list:
        for line in open(item_file_path):
            end_file.writelines(line)
        end_file.write('\n')
    # 关闭文件
    end_file.close()

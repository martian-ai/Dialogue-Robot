# coding:utf-8
# author:Apollo2Mars@gmail.com

import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import multiprocessing
import tensorflow as tf
import six

class Dataset_ChineseSTS():

    def __init__(self, tokenizer, max_seq_len, data_type, tag_list):
        self.tokenizer = tokenizer
        self.word2idx = self.tokenizer.word2idx
        self.max_seq_len = max_seq_len
        self.tag_list = tag_list
        self.data_type = data_type
        self.__set_tag2id()
        self.__set_tag2onehot()
        print("tag list", tag_list)
        print("tag to index", self.tag2idx)
        print("index to tag", self.idx2tag)

    def __set_tag2id(self):
        tag2idx = {}
        idx2tag = {}
        for idx, item in enumerate(self.tag_list):
            tag2idx[item] = idx
            idx2tag[idx] = item

        self.tag2idx = tag2idx
        self.idx2tag = idx2tag
 
    def __set_tag2onehot(self):
        tag_list = self.tag_list
        onehot_encoder = OneHotEncoder(sparse=False)
        one_hot_df = onehot_encoder.fit_transform(
            np.asarray(list(range(len(tag_list)))).reshape(-1,1))

        tag_dict = {}
        for aspect, vector in zip(tag_list, one_hot_df):
            tag_dict[aspect] = vector

        self.tag_dict_onehot = tag_dict

    def __pad_and_truncate(self, sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
        """
        :param sequence:
        :param maxlen:
        :param dtype:
        :param padding:
        :param truncating:
        :param value:
        :return: sequence after padding and truncate
        """
        x = (np.ones(maxlen) * value).astype(dtype)

        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]

        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def __encode_text_sequence(self, text, do_padding, do_reverse):
        """
        :param text:
        :return: convert text to numberical digital features with max length, paddding
        and truncating
        """
        words = list(text)

        sequence = [self.word2idx[w] if w in self.word2idx else self.word2idx['<UNK>'] for w in words]

        if len(sequence) == 0:
            sequence = [0]
        if do_reverse:
            sequence = sequence[::-1]

        if do_padding:
            sequence = self.__pad_and_truncate(sequence, self.max_seq_len, value=self.word2idx["<PAD>"])

        return sequence

    def del_unbalance_label(self):
        pass

    def visualization(self):
        pass

    def preprocess(self, corpus):
        fin = open(corpus, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        text_a, text_b, score = [], [], []
        for line in lines:
            line = line.strip('\t\n\r')
            cut_list = line.split('\t')
            if len(cut_list) == 5:
                text_a.append(cut_list[1])
                text_b.append(cut_list[3])
                score.append(cut_list[4])
            else:
                print("error line", line)
                raise Exception("Raise Exception")

        instances = []
        for tmp_a, tmp_b, tmp_score in zip(text_a, text_b, score):
            tmp_a = self.__encode_text_sequence(tmp_a, True, False)
            tmp_b = self.__encode_text_sequence(tmp_b, True, False)
            tmp_score = self.tag_dict_onehot[tmp_score]
            instances.append({"a":tmp_a.tolist(), "b":tmp_b.tolist(), 'score':tmp_score.tolist()})

        return instances

# class BatchGenerator(object):
#     def __init__(self, instances, batch_size=32, training=False, num_parallel_calls=0, shuffle_ratio=1.0):
#         self.instances = instances
#         self.batch_size = batch_size
#         self.training = training
#         self.shuffle_ratio = shuffle_ratio
#         self.num_parallel_calls = num_parallel_calls if num_parallel_calls>0 else multiprocessing.cpu_count()/2

#         if self.instances is None or len(self.instances) == 0:
#             raise ValueError('empty instances!!')

#         self.batch, self.init_op = self.build_input_pipeline()

#         sess_conf = tf.ConfigProto()
#         sess_conf.gpu_options.allow_growth = True
#         self.session = tf.Session(config=sess_conf)
#         self.session.run(tf.tables_initializer())

#     def next(self):
#         return self.session.run(self.batch)

#     def init(self):
#         self.session.run(self.init_op)

#     def get_instance_size(self):
#         return len(self.instances)

#     def get_batch_size(self):
#         return self.batch_size

#     def get_instances(self):
#         return self.instances

#     def build_input_pipeline(self):

#         def detect_input_type_shape(instance):
#             def get_type(value):
#                 print("*"*100)
#                 print(type(value))
#                 if isinstance(value, six.string_types):
#                     return tf.string
#                 elif isinstance(value, bool):
#                     return tf.bool
#                 elif isinstance(value, int):
#                     return tf.int32
#                 elif isinstance(value, float):
#                     return tf.float32
#                 elif isinstance(value, list):
#                     print("A"*100)
#                     return get_type(value[0])
#                 else:
#                     print("B"*100)
#                     return None
            
#             def get_shape(value):
#                 if not isinstance(value, list):
#                     return tf.TensorShape([])
#                 first_dim = len(value) if isinstance(value, list) else 1
#                 second_dim = len(value[0]) if isinstance(value[0], list) else 1
#                 # third_dim = len(value[0][0])

#                 # 当前可以遇到的情况共有 [8], [8,8,8], [[8,8],[8,8]], [[1,2,3]]
#                 # first dim           1    3        2               1    
#                 # seconde dim         1    1        2               3
#                 # third dim           1    1        1               1

#                 if first_dim == 1 and second_dim == 1:
#                     return tf.TensorShape([None]) # [8]
#                 elif first_dim != 1 and second_dim == 1:
#                     return tf.TensorShape([None]) # [8,8,8]
#                 elif first_dim == 1 and second_dim != 1:
#                     return tf.TensorShape([None, None]) # [[1,2,3]]
#                 elif first_dim != 1 and second_dim != 1:
#                     return tf.TensorShape([None, None]) # [[2,3], [3,4]]
#                 else:
#                     return tf.TensorShape([None, None, None]) # TODO
            
#             fields = instance.keys()
#             input_type = {}
#             input_shape = {}

#             for field in fields:
#                 field_type = get_type(instance[field])
#                 field_shape = get_shape(instance[field])
#                 input_type[field] = field_type
#                 input_shape[field] = field_shape

#             return fields, input_type, input_shape

#         def make_generator():
#             for instance in self.instances:
#                 yield instance

#         def build_padded_shape(output_shapes):
#             padded_shapes = dict()
#             for field, shape in output_shapes.items():
#                 field_dim = len(shape.as_list())
#                 if field_dim > 0:
#                     padded_shapes[field] = tf.TensorShape([None] * field_dim)
#                 else:
#                     padded_shapes[field] = tf.TensorShape([])
#             return padded_shapes

#         fields, type_dict, shape_dict = detect_input_type_shape(self.instances[0])

#         dataset = tf.data.Dataset.from_generator(make_generator,
#                                                  {w: type_dict[w] for w in fields}, 
#                                                  {w: shape_dict[w] for w in fields})

#         # dataset = dataset.map(lambda x : transform_new_instance(x))
#         # 处理变长数据，当前数据已经padding, 故不需此步骤
#         # TODO 变长数据 使用tf.Dataset 处理方法
#         dataset = dataset.prefetch(self.batch_size)
#         padded_shapes = build_padded_shape(dataset.output_shapes)
#         dataset = dataset.padded_batch(self.batch_size, padded_shapes=padded_shapes)
#         iterator = dataset.make_initializable_iterator()
#         init_op = iterator.initializer
#         output_dict = iterator.get_next()
#         return output_dict, init_op

if __name__ == "__main__":
    import sys
    sys.path.append('/Users/sunhongchao/Documents/craft/06_NLU')
    from solutions.utils.tokenizer.Tokenizer import build_tokenizer
    corpus_file = '/Users/sunhongchao/Documents/craft/06_NLU/resources/corpus/h_retrieval/similarity/ChineseSTS/simtrain_to05sts.txt'
    tokenizer = build_tokenizer([corpus_file], 'chineseSTS', 'Retrieval', 'tencent')
    dataset = Dataset_ChineseSTS(tokenizer, 16, 'retrieval', ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0'])
    instances = dataset.preprocess('/Users/sunhongchao/Documents/craft/06_NLU/resources/corpus/h_retrieval/similarity/ChineseSTS/simtrain_to05sts.txt')
    # generator = BatchGenerator(instances)
    # generator.init()
    # print(generator.next())

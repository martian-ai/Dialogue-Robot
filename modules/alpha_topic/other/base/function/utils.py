import numpy as np
import jieba
from collections import Counter, defaultdict

from six.moves import xrange


def sampling_from_dist(prob):
    """ Sample index from a list of unnormalised probability distribution
        same as np.random.multinomial(1, prob/np.sum(prob)).argmax()

    Parameters
    ----------
    prob: ndarray
        array of unnormalised probability distribution

    Returns
    -------
    new_topic: return a sampled index
    """
    thr = prob.sum() * np.random.rand()
    new_topic = 0
    tmp = prob[new_topic]
    while tmp < thr:
        new_topic += 1
        # print('prob', prob)
        # print('new_topic', new_topic)
        tmp += prob[new_topic]
    return new_topic


def sampling_from_dict(prob):
    """ sample key from dictionary `prob` where values are unnormalised probability distribution

    Parameters
    ----------
    prob: dict
        key = topic
        value = unnormalised probability of the topic

    Returns
    -------
    key: int
        sampled key
    """
    thr = np.random.rand() * sum(prob.values())

    tmp = 0
    new_topic = ''
    for key, p in prob.items():
        tmp += p
        if tmp < thr:
            new_topic = key
    return new_topic


def isfloat(value):
    """
    Check the value is convertable to float value
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def read_voca(path):
    """
    open file from path and read each line to return the word list
    """
    with open(path, 'r') as f:
        return [word.strip() for word in f.readlines()]


def word_cnt_to_bow_list(word_ids, word_cnt):
    corpus_list = list()
    for di in xrange(len(word_ids)):
        doc_list = list()
        for wi in xrange(len(word_ids[di])):
            word = word_ids[di][wi]
            for c in xrange(word_cnt[di][wi]):
                doc_list.append(word)
        corpus_list.append(doc_list)
    return corpus_list


def log_normalize(log_prob_vector):
    """
    returns a probability vector of log probability vector
    """
    max_v = log_prob_vector.max()
    log_prob_vector += max_v
    log_prob_vector = np.exp(log_prob_vector)
    log_prob_vector /= log_prob_vector.sum()
    return log_prob_vector


def convert_cnt_to_list(word_ids, word_cnt):
    corpus = list()

    for di in xrange(len(word_ids)):
        doc = list()
        doc_ids = word_ids[di]
        doc_cnt = word_cnt[di]
        for wi in xrange(len(doc_ids)):
            word_id = doc_ids[wi]
            for si in xrange(doc_cnt[wi]):
                doc.append(word_id)
        corpus.append(doc)
    return corpus


def write_top_words(topic_word_matrix,
                    vocab,
                    filepath,
                    n_words=20,
                    delimiter=',',
                    newline='\n'):
    with open(filepath, 'w') as f:
        for ti in xrange(topic_word_matrix.shape[0]):
            top_words = vocab[topic_word_matrix[ti, :].argsort()[::-1]
                              [:n_words]]
            f.write('%d' % (ti))
            for word in top_words:
                f.write(delimiter + word)
            f.write(newline)


def get_top_words(topic_word_matrix, vocab, topic, n_words=20):
    if not isinstance(vocab, np.ndarray):
        vocab = np.array(vocab)
    top_words = vocab[topic_word_matrix[topic].argsort()[::-1][:n_words]]
    return top_words


def get_ids_cnt(corpus, voca, max_voca=9999999, remove_top_n=0):
    """

    Returns
    -------
    voca_list: ndarray
        list of vocabulary used to construct a corpus
    doc_ids: list
        list of list of word id for each document
    doc_cnt: list
        list of list of word count for each document
    """
    stop = []
    docs = list()
    freq = Counter()

    for doc in corpus:
        doc = [
            word.lower() for word in doc
            if word.lower() in voca and word.lower() not in stop
        ]
        # print(doc)
        # doc = [
        #     word.lower() for word in doc if word.lower() in voca
        #     and word.lower() not in stop and len(word) != 1
        # ]
        freq.update(doc)
        docs.append(doc)

    voca = [
        key for iter, (key, val) in enumerate(freq.most_common(max_voca))
        if iter >= remove_top_n
    ]
    voca_dic = dict()
    voca_list = list()
    for word in voca:
        voca_dic[word] = len(voca_dic)
        voca_list.append(word)
    doc_ids = list()
    doc_cnt = list()
    for doc in docs:
        words = set(doc)
        ids = np.array(
            [int(voca_dic[word]) for word in words if word in voca_dic])
        cnt = np.array(
            [int(doc.count(word)) for word in words if word in voca_dic])
        doc_ids.append(ids)
        doc_cnt.append(cnt)
    return np.array(voca_list), doc_ids, doc_cnt


def get_corpus_ids_cnt(corpus_path='corpus.txt',
                       vocab_path='vocab.txt',
                       stopwords_path='stop.txt',
                       num_doc=100,
                       max_voca=10000,
                       remove_top_n=5):
    """To get test data for training a model
    reuters, stopwords, english words corpora should be installed in nltk_data: nltk.download()

    Parameters
    ----------
    corpus_path: str
        corpus path for training data
    voacab_path: str
        vocab path 
    num_doc: int
        number of documents to be returned
    max_voca: int
        maximum number of vocabulary size for the returned corpus
    remove_top_n: int
        remove top n frequently used words

    """

    lines = []
    with open(corpus_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [item.strip().split("\t")[1] for item in lines]
    vocab = []
    with open(vocab_path, mode='r', encoding='utf-8') as f:
        for item in f.readlines():
            vocab.append(item.strip('\n').split('\t')[0])
    stopwords = []
    with open(stopwords_path, encoding='utf-8', mode='r') as f:
        for item in f.readlines():
            stopwords.append(item.strip())
    corpus = []
    for item in lines[:num_doc]:
        corpus.append(
            [tmp for tmp in list(jieba.cut(item)) if tmp not in stopwords])

    return lines, get_ids_cnt(corpus, vocab, max_voca, remove_top_n)

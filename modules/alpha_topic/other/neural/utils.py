# -*- coding: utf-8 -*-
# 基础utils 类
# 包括 logging, distirbuted, reporter, trainer

from __future__ import print_function
from __future__ import absolute_import

import os
import time
import math
import sys
import torch
import logging
import math
import pickle
import torch.distributed

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from datetime import datetime
# from models.loss import LogisticLossCompute, abs_loss
from tensorboardX import SummaryWriter


logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


""" Pytorch Distributed utils
    This piece of code was heavily inspired by the equivalent of Fairseq-py
    https://github.com/pytorch/fairseq
"""


def is_master(gpu_ranks, device_id):
    return gpu_ranks[device_id] == 0


def multi_init(device_id, world_size, gpu_ranks):
    print(gpu_ranks)
    dist_init_method = 'tcp://localhost:10000'
    dist_world_size = world_size
    torch.distributed.init_process_group(backend='nccl',
                                         init_method=dist_init_method,
                                         world_size=dist_world_size,
                                         rank=gpu_ranks[device_id])
    gpu_rank = torch.distributed.get_rank()
    if not is_master(gpu_ranks, device_id):
        logger.disabled = True

    return gpu_rank


def all_reduce_and_rescale_tensors(tensors,
                                   rescale_denom,
                                   buffer_size=10485760):
    """All-reduce and rescale tensors in chunks of the specified size.

    Args:
        tensors: list of Tensors to all-reduce
        rescale_denom: denominator for rescaling summed Tensors
        buffer_size: all-reduce chunk size in bytes
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    buffer_t = tensors[0].new(
        math.ceil(buffer_size / tensors[0].element_size())).zero_()
    buffer = []

    def all_reduce_buffer():
        # copy tensors into buffer_t
        offset = 0
        for t in buffer:
            numel = t.numel()
            buffer_t[offset:offset + numel].copy_(t.view(-1))
            offset += numel

        # all-reduce and rescale
        torch.distributed.all_reduce(buffer_t[:offset])
        buffer_t.div_(rescale_denom)

        # copy all-reduced buffer back into tensors
        offset = 0
        for t in buffer:
            numel = t.numel()
            t.view(-1).copy_(buffer_t[offset:offset + numel])
            offset += numel

    filled = 0
    for t in tensors:
        sz = t.numel() * t.element_size()
        if sz > buffer_size:
            # tensor is bigger than buffer, all-reduce and rescale directly
            torch.distributed.all_reduce(t)
            t.div_(rescale_denom)
        elif filled + sz > buffer_size:
            # buffer is full, all-reduce and replace buffer with grad
            all_reduce_buffer()
            buffer = [t]
            filled = sz
        else:
            # add tensor to buffer
            buffer.append(t)
            filled += sz

    if len(buffer) > 0:
        all_reduce_buffer()


def all_gather_list(data, max_size=4096):
    """Gathers arbitrary data from all nodes into a list."""
    world_size = torch.distributed.get_world_size()
    if not hasattr(all_gather_list, '_in_buffer') or \
            max_size != all_gather_list._in_buffer.size():
        all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
        all_gather_list._out_buffers = [
            torch.cuda.ByteTensor(max_size) for i in range(world_size)
        ]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size +
                                                                    2))
    assert max_size < 255 * 256
    in_buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    in_buffer[1] = enc_size % 255
    in_buffer[2:enc_size + 2] = torch.ByteTensor(list(enc))

    torch.distributed.all_gather(out_buffers, in_buffer.cuda())

    results = []
    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (255 * out_buffer[0].item()) + out_buffer[1].item()

        bytes_list = bytes(out_buffer[2:size + 2].tolist())
        result = pickle.loads(bytes_list)
        results.append(result)
    return results


def build_report_manager(opt):
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(opt.tensorboard_log_dir +
                               datetime.now().strftime("/%b-%d_%H-%M-%S"),
                               comment="Unmt")
    else:
        writer = None

    report_mgr = ReportMgr(opt.report_every,
                           start_time=-1,
                           tensorboard_writer=writer)
    return report_mgr


class ReportMgrBase(object):
    """
    Report Manager Base class
    Inherited classes should override:
        * `_report_training`
        * `_report_step`
    """

    def __init__(self, report_every, start_time=-1.):
        """
        Args:
            report_every(int): Report status every this many sentences
            start_time(float): manually set report start time. Negative values
                means that you will need to set it later or use `start()`
        """
        self.report_every = report_every
        self.progress_step = 0
        self.start_time = start_time

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(self,
                        step,
                        num_steps,
                        learning_rate,
                        report_stats,
                        multigpu=False):
        """
        This is the user-defined batch-level traing progress
        report function.

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started
                                (set 'start_time' or use 'start()'""")

        if multigpu:
            report_stats = Statistics.all_gather_stats(report_stats)

        if step % self.report_every == 0:
            self._report_training(step, num_steps, learning_rate, report_stats)
            self.progress_step += 1
        return Statistics()

    def _report_training(self, *args, **kwargs):
        """ To be overridden """
        raise NotImplementedError()

    def report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        Report stats of a step

        Args:
            train_stats(Statistics): training stats
            valid_stats(Statistics): validation stats
            lr(float): current learning rate
        """
        self._report_step(lr,
                          step,
                          train_stats=train_stats,
                          valid_stats=valid_stats)

    def _report_step(self, *args, **kwargs):
        raise NotImplementedError()


class ReportMgr(ReportMgrBase):

    def __init__(self, report_every, start_time=-1., tensorboard_writer=None):
        """
        A report manager that writes statistics on standard output as well as
        (optionally) TensorBoard

        Args:
            report_every(int): Report status every this many sentences
            tensorboard_writer(:obj:`tensorboard.SummaryWriter`):
                The TensorBoard Summary writer to use or None
        """
        super(ReportMgr, self).__init__(report_every, start_time)
        self.tensorboard_writer = tensorboard_writer

    def maybe_log_tensorboard(self, stats, prefix, learning_rate, step):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard(prefix, self.tensorboard_writer,
                                  learning_rate, step)

    def _report_training(self, step, num_steps, learning_rate, report_stats):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        report_stats.output(step, num_steps, learning_rate, self.start_time)

        # Log the progress using the number of batches on the x-axis.
        self.maybe_log_tensorboard(report_stats, "progress", learning_rate,
                                   step)
        report_stats = Statistics()

        return report_stats

    def _report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        if train_stats is not None:
            self.log('Train perplexity: %g' % train_stats.ppl())
            self.log('Train accuracy: %g' % train_stats.accuracy())

            self.maybe_log_tensorboard(train_stats, "train", lr, step)

        if valid_stats is not None:
            self.log('Validation perplexity: %g' % valid_stats.ppl())
            self.log('Validation accuracy: %g' % valid_stats.accuracy())

            self.maybe_log_tensorboard(valid_stats, "valid", lr, step)


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self,
                 loss=0,
                 n_words=0,
                 n_correct=0,
                 rl_loss=0,
                 topic_loss=0):
        self.loss = loss
        self.n_words = n_words
        self.n_docs = 0
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()
        self.rl_loss = rl_loss
        self.topic_loss = topic_loss

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        from torch.distributed import get_rank
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.n_docs += stat.n_docs
        self.rl_loss += stat.rl_loss
        self.topic_loss += stat.topic_loss

        if update_n_src_words:
            self.n_src_words += stat.n_src_words

    def accuracy(self):
        """ compute accuracy """
        if self.n_words != 0:
            return 100 * (self.n_correct / self.n_words)
        else:
            return -1

    def xent(self):
        """ compute cross entropy """
        if self.n_words != 0:
            return self.loss / self.n_words
        else:
            return -1

    def ppl(self):
        """ compute perplexity """
        if self.n_words != 0:
            return math.exp(min(self.loss / self.n_words, 100))
        else:
            return -1

    def rlloss(self):
        return self.rl_loss

    def tploss(self):
        return self.topic_loss

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        logger.info((
            "Step %2d/%5d; acc: %6.2f; ppl: %5.2f; xent: %4.2f; rlloss: %4.2f; "
            + "tploss: %4.2f; lr: %7.8f; %3.0f/%3.0f tok/s; %6.0f sec") %
                    (step, num_steps, self.accuracy(), self.ppl(), self.xent(),
                     self.rlloss(), self.tploss(), learning_rate,
                     self.n_src_words / (t + 1e-5), self.n_words /
                     (t + 1e-5), time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/tploss", self.tploss(), step)
        writer.add_scalar(prefix + "/rlloss", self.rlloss(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)


class VocabWrapper(object):
    # Glove has not been implemented.
    def __init__(self, mode="word2vec", emb_size=100):
        self.mode = mode
        self.emb_size = emb_size
        self.model = None
        self.emb = None

    def _glove_init(self):
        pass

    def _word2vec_init(self):
        self.model = Word2Vec(size=self.emb_size,
                              window=5,
                              min_count=30,
                              workers=4)

    def _glove_train(self, ex):
        pass

    def _word2vec_train(self, ex):
        if self.model.wv.vectors.size == 0:
            self.model.build_vocab(ex, update=False)
        else:
            self.model.build_vocab(ex, update=True)
        self.model.train(ex, total_examples=self.model.corpus_count, epochs=1)

    def _glove_report(self):
        pass

    def _word2vec_report(self):
        if self.model is not None:
            print("Total examples: %d" % self.model.corpus_count)
            print("Vocab Size: %d" % len(self.model.wv.vocab))
        else:
            print("Vocab Size: %d" % len(self.emb.vocab))

    def _glove_save_model(self, path):
        pass

    def _word2vec_save_model(self, path):
        self._word2vec_report()
        self.model.save(path)

    def _glove_load_model(self, path):
        pass

    def _word2vec_load_model(self, path):
        self.model = Word2Vec.load(path)

    def _glove_save_emb(self, path):
        pass

    def _word2vec_save_emb(self, path):
        self._word2vec_report()
        if self.model is not None:
            self.model.wv.save(path)
        else:
            self.emb.save(path)

    def _glove_load_emb(self, path):
        pass

    def _word2vec_load_emb(self, path):
        self.emb = KeyedVectors.load(path)
        self.emb_size = self.emb.vector_size

    def _w2i_glove(self, w):
        return None

    def _w2i_word2vec(self, w):
        if self.emb is not None:
            if w in self.emb.vocab.keys():
                return self.emb.vocab[w].index
        if self.model is not None:
            if w in self.model.wv.vocab.keys():
                return self.model.wv.vocab[w].index
        return None

    def _i2w_glove(self, idx):
        return None

    def _i2w_word2vec(self, idx):
        if self.emb is not None:
            if idx < len(self.emb.vocab):
                return self.emb.index2word[idx]
        if self.model is not None:
            if idx < len(self.model.wv.vocab):
                return self.model.wv.index2word[idx]
        return None

    def _i2e_glove(self, idx):
        return None

    def _i2e_word2vec(self, idx):
        if self.emb is not None:
            if idx < len(self.emb.vocab):
                return self.emb.vectors[idx]
        if self.model is not None:
            if idx < len(self.model.wv.vocab):
                return self.model.wv.vectors[idx]
        return None

    def _w2e_glove(self, w):
        return None

    def _w2e_word2vec(self, w):
        if self.emb is not None:
            if w in self.emb.vocab.keys():
                return self.emb[w]
        if self.model is not None:
            if w in self.model.wv.vocab.keys():
                return self.model.wv[w]
        return None

    def _voc_size_glove(self):
        return -1

    def _voc_size_word2vec(self):
        if self.emb is not None:
            return len(self.emb.vocab)
        if self.model is not None:
            return len(self.model.wv.vocab)
        return -1

    def _get_emb_glove(self):
        return None

    def _get_emb_word2vec(self):
        if self.emb is not None:
            return self.emb.vectors
        if self.model is not None:
            return self.model.wv.vectors
        return None

    def init_model(self):
        if self.mode == "glove":
            self._glove_init()
        else:
            self._word2vec_init()

    def train(self, ex):
        """
        ex: training examples.
            [['我', '爱', '中国', '。'],
             ['这', '是', '一个', '句子', '。']]
        """
        if self.mode == "glove":
            self._glove_train(ex)
        else:
            self._word2vec_train(ex)

    def report(self):
        if self.mode == "glove":
            self._glove_report()
        else:
            self._word2vec_report()

    def save_model(self, path):
        if self.mode == "glove":
            self._glove_save_model(path)
        else:
            self._word2vec_save_model(path)

    def load_model(self, path):
        if self.mode == "glove":
            self._glove_load_model(path)
        else:
            self._word2vec_load_model(path)

    def save_emb(self, path):
        if self.mode == "glove":
            self._glove_save_emb(path)
        else:
            self._word2vec_save_emb(path)

    def load_emb(self, path):
        if self.mode == "glove":
            self._glove_load_emb(path)
        else:
            self._word2vec_load_emb(path)

    def w2i(self, w):
        if self.mode == "glove":
            return self._w2i_glove(w)
        else:
            return self._w2i_word2vec(w)

    def i2w(self, idx):
        if self.mode == "glove":
            return self._i2w_glove(idx)
        else:
            return self._i2w_word2vec(idx)

    def w2e(self, w):
        if self.mode == "glove":
            return self._w2e_glove(w)
        else:
            return self._w2e_word2vec(w)

    def i2e(self, idx):
        if self.mode == "glove":
            return self._i2e_glove(idx)
        else:
            return self._i2e_word2vec(idx)

    def voc_size(self):
        if self.mode == "glove":
            return self._voc_size_glove()
        else:
            return self._voc_size_word2vec()

    def get_emb(self):
        if self.mode == "glove":
            return self._get_emb_glove()
        else:
            return self._get_emb_word2vec()


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optims, tokenizer):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every,
                               start_time=-1,
                               tensorboard_writer=writer)

    trainer = Trainer(args, model, optims, tokenizer, grad_accum_count, n_gpu,
                      gpu_rank, report_manager)

    # print(tr)
    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self,
                 args,
                 model,
                 optims,
                 tokenizer,
                 grad_accum_count=1,
                 n_gpu=1,
                 gpu_rank=1,
                 report_manager=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optims = optims
        self.tokenizer = tokenizer
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

    def train(self,
              train_iter_fct,
              train_steps,
              valid_iter_fct=None,
              valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        step = self.optims[0]._step + 1
        true_batchs = []
        accum = 0
        examples = 0

        train_iter = train_iter_fct()
        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:

            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    true_batchs.append(batch)
                    examples += batch.tgt.size(0)
                    accum += 1
                    if accum == self.grad_accum_count:
                        if self.n_gpu > 1:
                            examples = sum(
                                distributed.all_gather_list(examples))

                        self._gradient_calculation(true_batchs, examples,
                                                   total_stats, report_stats,
                                                   step)

                        report_stats = self._maybe_report_training(
                            step, train_steps, self.optims[0].learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        examples = 0
                        if (step % self.save_checkpoint_steps == 0
                                and self.gpu_rank == 0):
                            self._save(step)
                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

    def _gradient_calculation(self, true_batchs, examples, total_stats,
                              report_stats, step):
        self.model.zero_grad()

        for batch in true_batchs:
            loss = self.model(batch)

            # Topic Model loss
            topic_stats = Statistics(topic_loss=loss.clone().item() /
                                     float(examples))
            loss.div(float(examples)).backward(retain_graph=False)
            total_stats.update(topic_stats)
            report_stats.update(topic_stats)

        if step % 1000 == 0:
            for k in range(self.args.topic_num):
                logger.info(','.join([
                    self.model.voc_id_wrapper.i2w(i)
                    for i in self.model.topic_model.tm1.beta.topk(20, dim=-1)
                    [1][k].tolist()
                ]))
        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.n_gpu > 1:
            grads = [
                p.grad.data for p in self.model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            distributed.all_reduce_and_rescale_tensors(grads, float(1))
        for o in self.optims:
            o.step()

    def _save(self, step):
        real_model = self.model

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optims': self.optims,
        }
        checkpoint_path = os.path.join(self.args.model_path,
                                       'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(step,
                                                       num_steps,
                                                       learning_rate,
                                                       report_stats,
                                                       multigpu=self.n_gpu > 1)

    def _report_step(self,
                     learning_rate,
                     step,
                     train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(learning_rate,
                                                   step,
                                                   train_stats=train_stats,
                                                   valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)


"""  train abstractive  """

#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time
import torch
import distributed

from pytorch_transformers import BertTokenizer
from models import data_loader
from models.data_loader import load_dataset
from models.optimizers import build_optim, build_optim_bert, build_optim_other, build_optim_topic
from models.rl_model import Model as Summarizer
from models.rl_predictor import build_predictor
from models.rl_model_trainer import build_trainer
from utils import logger, init_logger
from utils import rouge_results_to_str, test_bleu, test_length
from rouge import Rouge, FilesRouge

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_multi(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ run process """

    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_single(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def baseline(args, cal_lead=False, cal_oracle=False):
    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, args.test_batch_ex_size, 'cpu',
                                       shuffle=False, is_test=True)

    if cal_lead:
        mode = "lead"
    else:
        mode = "oracle"

    rouge = Rouge()
    pred_path = '%s.%s.pred' % (args.result_path, mode)
    gold_path = '%s.%s.gold' % (args.result_path, mode)
    save_pred = open(pred_path, 'w', encoding='utf-8')
    save_gold = open(gold_path, 'w', encoding='utf-8')

    with torch.no_grad():
        count = 0
        for batch in test_iter:
            summaries = batch.tgt_txt
            origin_sents = batch.original_str
            ex_segs = batch.ex_segs
            ex_segs = [sum(ex_segs[:i]) for i in range(len(ex_segs)+1)]

            for idx in range(len(summaries)):
                summary = " ".join(summaries[idx][1:-1]).replace("\n", "")
                txt = origin_sents[ex_segs[idx]:ex_segs[idx+1]]
                if cal_oracle:
                    selected = []
                    max_rouge = 0.
                    while True:
                        cur_max_rouge = max_rouge
                        cur_id = -1
                        for i in range(len(txt)):
                            if (i in selected):
                                continue
                            c = selected + [i]
                            temp_txt = " ".join([txt[j][9:] for j in c])
                            if len(temp_txt.split()) > args.ex_max_token_num:
                                continue
                            rouge_score = rouge.get_scores(temp_txt, summary)
                            rouge_1 = rouge_score[0]["rouge-1"]["f"]
                            rouge_l = rouge_score[0]["rouge-l"]["f"]
                            rouge_score = rouge_1 + rouge_l
                            if rouge_score > cur_max_rouge:
                                cur_max_rouge = rouge_score
                                cur_id = i
                        if (cur_id == -1):
                            break
                        selected.append(cur_id)
                        max_rouge = cur_max_rouge
                    pred_txt = " ".join([txt[j][9:] for j in selected])
                else:
                    k = min(args.ex_max_k, len(txt))
                    pred_txt = " ".join(txt[:k]).replace("\n", "")
                save_gold.write(summary + "\n")
                save_pred.write(pred_txt + "\n")
                count += 1
                if count % 10 == 0:
                    print(count)
    save_gold.flush()
    save_pred.flush()
    save_gold.close()
    save_pred.close()

    length = test_length(pred_path, gold_path)
    bleu = test_bleu(pred_path, gold_path)
    file_rouge = FilesRouge(hyp_path=pred_path, ref_path=gold_path)
    pred_rouges = file_rouge.get_scores(avg=True)
    logger.info('Length ratio:\n%s' % str(length))
    logger.info('Bleu:\n%s' % str(bleu*100))
    logger.info('Rouges:\n%s' % rouge_results_to_str(pred_rouges))


def validate(args, device_id):
    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        best_dev_steps = -1
        best_dev_results = (0, 0)
        best_test_results = (0, 0)
        patient = 100
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            if (args.test_start_from != -1 and step < args.test_start_from):
                xent_lst.append((1e6, cp))
                continue
            logger.info("Step %d: processing %s" % (i, cp))
            rouge_dev = validate_(args, device_id, cp, step)
            rouge_test = test(args, device_id, cp, step)
            if (rouge_dev[0] + rouge_dev[1]) > (best_dev_results[0] + best_dev_results[1]):
                best_dev_results = rouge_dev
                best_test_results = rouge_test
                best_dev_steps = step
                patient = 100
            else:
                patient -= 1
            logger.info("Current step: %d" % step)
            logger.info("Dev results: ROUGE-1-l: %f, %f" % (rouge_dev[0], rouge_dev[1]))
            logger.info("Test results: ROUGE-1-l: %f, %f" % (rouge_test[0], rouge_test[1]))
            logger.info("Best step: %d" % best_dev_steps)
            logger.info("Best dev results: ROUGE-1-l: %f, %f" % (best_dev_results[0], best_dev_results[1]))
            logger.info("Best test results: ROUGE-1-l: %f, %f\n\n" % (best_test_results[0], best_test_results[1]))

            if patient == 0:
                break

    else:
        best_dev_results = (0, 0)
        best_test_results = (0, 0)
        best_dev_steps = -1
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    rouge_dev = validate_(args, device_id, cp, step)
                    rouge_test = test(args, device_id, cp, step)
                    if (rouge_dev[0] + rouge_dev[1]) > (best_dev_results[0] + best_dev_results[1]):
                        best_dev_results = rouge_dev
                        best_test_results = rouge_test
                        best_dev_steps = step

                    logger.info("Current step: %d" % step)
                    logger.info("Dev results: ROUGE-1-l: %f, %f" % (rouge_dev[0], rouge_dev[1]))
                    logger.info("Test results: ROUGE-1-l: %f, %f" % (rouge_test[0], rouge_test[1]))
                    logger.info("Best step: %d" % best_dev_steps)
                    logger.info("Best dev results: ROUGE-1-l: %f, %f" % (best_dev_results[0], best_dev_results[1]))
                    logger.info("Best test results: ROUGE-1-l: %f, %f\n\n" % (best_test_results[0], best_test_results[1]))

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)


def validate_(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    model = Summarizer(args, device, tokenizer.vocab, checkpoint)
    model.eval()

    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'dev', shuffle=False),
                                        args.test_batch_size, args.test_batch_ex_size, device,
                                        shuffle=False, is_test=True)

    predictor = build_predictor(args, tokenizer, model, logger)
    rouge = predictor.validate(valid_iter, step)
    return rouge


def test(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    model = Summarizer(args, device, tokenizer.vocab, checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, args.test_batch_ex_size, device,
                                       shuffle=False, is_test=True)

    predictor = build_predictor(args, tokenizer, model, logger)
    rouge = predictor.validate(test_iter, step)
    return rouge


def test_text(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    model = Summarizer(args, device, tokenizer.vocab, checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, args.test_batch_ex_size, device,
                                       shuffle=False, is_test=True)
    predictor = build_predictor(args, tokenizer, model, logger)
    predictor.translate(test_iter, step)


def train(args, device_id):
    if (args.world_size > 1):
        train_multi(args)
    else:
        train_single(args, device_id)


def train_single(args, device_id):
    init_logger(args.log_file)
    logger.info(str(args))
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True),
                                      args.batch_size, args.batch_ex_size, device,
                                      shuffle=True, is_test=False)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    model = Summarizer(args, device, tokenizer.vocab, checkpoint)

    if args.train_from_ignore_optim:
        checkpoint = None
    if args.sep_optim:
        if args.encoder == 'bert':
            optim_bert = build_optim_bert(args, model, checkpoint)
            optim_other = build_optim_other(args, model, checkpoint)
            if args.topic_model:
                optim_topic = build_optim_topic(args, model, checkpoint)
                optim = [optim_bert, optim_other, optim_topic]
            else:
                optim = [optim_bert, optim_other]
        else:
            optim_other = build_optim_other(args, model, checkpoint)
            if args.topic_model:
                optim_topic = build_optim_topic(args, model, checkpoint)
                optim = [optim_other, optim_topic]
            else:
                optim = [optim_other]
    else:
        optim = [build_optim(args, model, checkpoint, args.warmup)]

    logger.info(model)

    trainer = build_trainer(args, device_id, model, optim, tokenizer)

    if args.pretrain:
        trainer.train(train_iter_fct, args.pretrain_steps)
    else:
        trainer.train(train_iter_fct, args.train_steps)
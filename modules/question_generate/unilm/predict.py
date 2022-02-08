import torch
from modules.alpha_nn.modeling_bert import BertForSeq2SeqDecoder
from modules.alpha_nn.function.optimization import warmup_linear
from modules.utils.tokenizer.tokenization import BertTokenizer
import modules.question_generate.unilm.function.seq2seq_loader as seq2seq_loader
from modules.question_generate.unilm.function.decode_seq2seq import detokenize


class QG_Unilm_Full():
    def __init__(self, model_state_dict) -> None:
        no_cuda = True  # 默认预测不使用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('resources/embedding/bert-base-chinese/', do_lower_case=False)  # TODO, 当前原因是因为下载失败
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=False)
        cls_num_labels = 2
        pair_num_relation = 0
        type_vocab_size = 6  # 必须是6 否则报错

        mask_word_id, eos_word_ids, sos_word_id = self.tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]", "[S2S_SOS]"])
        self.model = BertForSeq2SeqDecoder.from_pretrained('resources/embedding/bert-base-chinese/',
                                                           state_dict=torch.load(model_state_dict, map_location='cpu'), num_labels=cls_num_labels,
                                                           num_rel=pair_num_relation, type_vocab_size=type_vocab_size,
                                                           task_idx=3, mask_word_id=mask_word_id, search_beam_size=1,
                                                           length_penalty=0, eos_id=eos_word_ids, sos_id=sos_word_id,
                                                           forbid_duplicate_ngrams=False, forbid_ignore_set=None,
                                                           not_predict_set=None, ngram_size=3, min_len=None, mode='s2s',
                                                           max_position_embeddings=512, ffn_type=0, num_qkv=0, seg_emb=False, pos_shift=False)
        self.model.eval()
        self.bi_uni_pipeline = []
        self.bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(list(self.tokenizer.vocab.keys()), self.tokenizer.convert_tokens_to_ids, 512,
                                                                             max_tgt_length=48, new_segment_ids='s2s', mode="s2s", num_qkv=0))

    def predict(self, doc: str, answer: str) -> str:
        input_lines = [" ".join(list(doc)) + " [SEP] " + " ".join(list(answer))]
        input_lines = [self.tokenizer.tokenize(x)[:512] for x in input_lines]
        input_lines = sorted(list(enumerate(input_lines)), key=lambda x: -len(x[1]))
        output_lines = [""] * len(input_lines)

        _chunk = input_lines[0:1]
        buf_id = [x[0] for x in _chunk]
        buf = [x[1] for x in _chunk]
        max_a_len = max([len(x) for x in buf])
        instances = []
        for instance in [(x, max_a_len) for x in buf]:
            for proc in self.bi_uni_pipeline:
                instances.append(proc(instance))
        with torch.no_grad():

            batch = seq2seq_loader.batch_list_to_batch_tensors(instances)
            batch = [t.to(self.device) if t is not None else None for t in batch]
            input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
            traces = self.model(input_ids, token_type_ids, position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)

            output_ids = traces.tolist()
            for i in range(len(buf)):
                w_ids = output_ids[i]
                output_buf = self.tokenizer.convert_ids_to_tokens(w_ids)
                output_tokens = []
                for t in output_buf:
                    if t in ("[SEP]", "[PAD]"):
                        break
                    output_tokens.append(t)
                output_sequence = ''.join(detokenize(output_tokens))
                output_lines[buf_id[i]] = output_sequence
        return output_lines[0]

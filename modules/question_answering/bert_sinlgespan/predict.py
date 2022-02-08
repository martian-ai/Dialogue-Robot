
import torch
from modules.alpha_nn.modeling_bert import BertConfig, BertForQuestionAnswering
from modules.utils.tokenizer.tokenization import BertTokenizer
from modules.question_answering.bert_sinlgespan.function.bert import evaluate, to_list # TODO
from modules.question_answering.bert_sinlgespan.function.utils_squad import RawResult, SquadExample, convert_examples_to_features, write_predictions
# from modules.basic_search.recall.function.recaller import EsRecaller
from torch.utils.data import DataLoader, TensorDataset

# from pytorch_transformers import (BertConfig, BertForQuestionAnswering,
#                                   BertTokenizer)


def build_dataset_example_feature_by_context_query(query, context, tokenizer, max_query_length, max_seq_length, doc_stride):
    examples = []
    recall_scores, contexts = [1.0], [context]
    for idx, context in enumerate(contexts):
        example = SquadExample(qas_id=idx, question_text=query, doc_tokens=list(context), orig_answer_text=None, start_position=None, end_position=None, is_impossible=False)
        examples.append(example)

    features = convert_examples_to_features(examples=examples, tokenizer=tokenizer, max_seq_length=max_seq_length, doc_stride=doc_stride, max_query_length=max_query_length, is_training=not evaluate)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index, all_cls_index, all_p_mask)
    return dataset, examples, features, recall_scores


# def build_dataset_example_feature(query, tokenizer, max_query_length, max_seq_length, doc_stride, search_size, es_index):
#     examples = []
#     recall_scores, contexts = get_search_results(query, search_size, es_index)
#     for idx, context in enumerate(contexts):
#         example = SquadExample(qas_id=idx, question_text=query, doc_tokens=list(context), orig_answer_text=None, start_position=None, end_position=None, is_impossible=False)
#         examples.append(example)

#     features = convert_examples_to_features(examples=examples, tokenizer=tokenizer, max_seq_length=max_seq_length, doc_stride=doc_stride, max_query_length=max_query_length, is_training=not evaluate)
#     all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
#     all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
#     all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
#     all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
#     all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
#     all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
#     dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index, all_cls_index, all_p_mask)
#     return dataset, examples, features, recall_scores


class QA_BERT_SingleSpan():
    def __init__(self, model_state_dict, config_file) -> None:
        no_cuda = True
        self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=False)
        # config = BertConfig.from_pretrained('bert-base-chinese')
        config = BertConfig(vocab_size_or_config_json_file=config_file)
        self.model = BertForQuestionAnswering(config)
        self.model.load_state_dict(torch.load(model_state_dict, map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()  # TODO

    # def predict_old(self, query, search_size, max_query_length, max_answer_length, max_seq_length, n_best_size, doc_stride, verbose_logging, es_index,  null_score_diff_threshold, prefix=""):
    #     dataset, examples, features, recall_scores = build_dataset_example_feature(query, self.tokenizer, max_query_length, max_seq_length, doc_stride, search_size, es_index)
    #     eval_dataloader = DataLoader(dataset, sampler=None, batch_size=16)
    #     all_results = []
    #     for batch in eval_dataloader:
    #         batch = tuple(t.to(self.device) for t in batch)
    #         with torch.no_grad():
    #             inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids':  batch[2]}
    #             example_indices = batch[3]
    #             outputs = self.model(**inputs)
    #         for i, example_index in enumerate(example_indices):
    #             eval_feature = features[example_index.item()]
    #             unique_id = int(eval_feature.unique_id)
    #             result = RawResult(unique_id=unique_id, start_logits=to_list(outputs[0][i]), end_logits=to_list(outputs[1][i]))
    #             all_results.append(result)
    #     all_predictions = write_predictions(examples, features, all_results, n_best_size, max_answer_length, True, None, None, None, verbose_logging, False, null_score_diff_threshold)
    #     return all_predictions, recall_scores

    def predict(self, doc=None, query=None):
        """
        function

        params
            doc : 输入文档
            query : 输入查询话术
        returns

        """

        if doc != None:
            pass
            # TODO
            # 插入文档
        else:
            dataset, examples, features, _ = build_dataset_example_feature_by_context_query(query, self.tokenizer, 16, 384, 128)

        eval_dataloader = DataLoader(dataset, sampler=None, batch_size=16)
        all_results = []
        for batch in eval_dataloader:
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
                example_indices = batch[3]
                outputs = self.model(**inputs)
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(unique_id=unique_id, start_logits=to_list(outputs[0][i]), end_logits=to_list(outputs[1][i]))
                all_results.append(result)
        return write_predictions(examples, features, all_results, 3, 24, True, None, None, None, False, False, 0.0)


# !TODO: add main test function

# if __name__ == "__main__":
#     no_cuda = True
#     model_state_dict = '../../../resources/models/mrc/pytorch_model.bin'
#     search_size = 3
#     max_query_length = 16 # Todo
#     max_answer_length = 24
#     max_seq_length = 384
#     doc_stride = 128
#     n_best_size = 3 # The total number of n-best predictions to generate in the nbest_predictions.json output file.
#     null_score_diff_threshold = 0.0 # "If true, all of the warnings related to data processing will be printed. "
#     device = torch.device( "cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
#     tokenizer = BertTokenizer.from_pretrained( 'bert-base-chinese', do_lower_case=False)
#     config = BertConfig.from_pretrained('bert-base-chinese')
#     model = BertForQuestionAnswering(config)
#     model.load_state_dict(torch.load(model_state_dict, map_location='cpu'))
#     model.to(device)
#     model.eval()

#     while True:
#         query = input("Please Enter:")
#         while not query:
#             print('Input should not be empty!')
#             query = input("Please Enter:")
#         (a,b), c = predict(query, model, 'bert-base-chinese', device, tokenizer, search_size, max_query_length, max_answer_length, max_seq_length, n_best_size, doc_stride, False, 'document-threebody', null_score_diff_threshold)
#         print(a)
#         print(b[0])
#         print(b[1])
#         print(b[2])
#         print(c)

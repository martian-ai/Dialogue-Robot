import json
import torch
import argparse
from pytorch_transformers import (BertConfig, BertForQuestionAnswering,
                                  BertTokenizer)
from bert_qa import evaluate
import os
import logging
from tqdm import tqdm
import jieba

from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions, write_predictions_extended)
from utils_squad import SquadExample
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

import os,sys
sys.path.append('../../../')

from examples.orqa.elastic_search import es_search

logger = logging.getLogger(__name__)


def to_list(tensor): # Tddo move to utils
    return tensor.detach().cpu().tolist()

def get_search_results(query, search_engine='offline-es'):
    if search_engine == 'offline-es':
        results_search = es_search(query, 'three-body')
        print(results_search)

        score_list, text_list = [], []
        for item in results_search['hits']['hits']:
            print(item['_score'])
            print(item['_source']['text'])
            score_list.append(item['_score'])
            text_list.append(item['_source']['text'])
        return score_list, text_list


def build_dataset_example_feature(args, query, tokenizer):

    examples = []
    scores, contexts = get_search_results(query, 'offline-es')

    for context in contexts:
        print("*"*39)
        print(context)
        example = SquadExample(
            qas_id=-1,
            question_text=query,
            doc_tokens=list(jieba.cut(context)), # Todo 
            orig_answer_text=None,
            start_position=None,
            end_position=None,
            is_impossible=False)

        print(example)
        examples.append(example)

    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=args.max_seq_length,
                                            doc_stride=args.doc_stride,
                                            max_query_length=args.max_query_length,
                                            is_training=not evaluate)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)

    return dataset, examples, features


def predict(args, query, model, tokenizer, prefix=""):
    dataset, examples, features = build_dataset_example_feature(args, query, tokenizer)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    #eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=None, batch_size=args.eval_batch_size)

    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch) 
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': None if args.model_type == 'xlm' else batch[2]  # XLM don't use segment_ids
                      }
            example_indices = batch[3]
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id    = unique_id,
                                start_logits = to_list(outputs[0][i]),
                                end_logits   = to_list(outputs[1][i]))
            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join('./', "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join('./', "nbest_predictions_{}.json".format(prefix))
    output_null_log_odds_file = None

    all_predictions = write_predictions(examples, features, all_results, args.n_best_size,
                        args.max_answer_length, args.do_lower_case, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                        args.version_2_with_negative, args.null_score_diff_threshold)
    
    return all_predictions


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: ")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ")

    ## Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name")


    parser.add_argument(
        '--version_2_with_negative',
        action='store_true',
        help='If true, the SQuAD examples contain some that do not have an answer.'
    )
    parser.add_argument(
        '--null_score_diff_threshold',
        type=float,
        default=0.0,
        help=
        "If null_score - best_non_null is greater than the threshold predict null."
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help=
        "The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded."
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help=
        "When splitting up a long document into chunks, how much stride to take between chunks."
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help=
        "The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.")
    parser.add_argument(
        "--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument(
        "--do_eval",
        action='store_true',
        help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training",
        action='store_true',
        help="Rul evaluation during training at each logging step.")
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Set this flag if you are using an uncased model.")

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight deay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help=
        "The total number of n-best predictions to generate in the nbest_predictions.json output file."
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help=
        "The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.")
    parser.add_argument(
        "--verbose_logging",
        action='store_true',
        help=
        "If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument(
        '--logging_steps', type=int, default=50, help="Log every X updates steps.")
    parser.add_argument(
        '--save_steps',
        type=int,
        default=50,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action='store_true',
        help=
        "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
    )
    parser.add_argument(
        "--no_cuda",
        action='store_true',
        help="Whether not to use CUDA when available")
    parser.add_argument(
        '--overwrite_cache',
        action='store_true',
        help="Overwrite the cached training and evaluation sets")
    parser.add_argument(
        '--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus")

    parser.add_argument(
        "--state_dict",
        default=None,
        type=str,
        required=True,
        help="model para after pretrained")

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device


    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-chinese', do_lower_case=False)

    config = BertConfig.from_pretrained('bert-base-chinese')
    model = BertForQuestionAnswering(config)
    model_state_dict = args.state_dict
    model.load_state_dict(torch.load(model_state_dict, map_location='cpu'))
    # map_location='cpu'
    model.to(args.device)
    model.eval()

    while True:
        query = input("Please Enter:")
        while not query:
            print('Input should not be empty!')
            query = input("Please Enter:")
        print(predict(args, query, model, tokenizer))

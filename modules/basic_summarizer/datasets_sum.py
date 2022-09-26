import pandas as pd
import datasets
from datasets import load_dataset, Dataset
from transformers import BertTokenizer
 
max_input_length = 512
max_target_length = 128
 
lcsts_part_1=pd.read_table('../../resources/corpus/LCSTS_ORIGIN/LCSTS_ORIGIN/DATA/PART_II.txt', header=None,
                           warn_bad_lines=True, error_bad_lines=False, sep='<[/d|/s|do|su|sh][^a].*>', encoding='utf-8')
lcsts_part_1=lcsts_part_1[0].dropna()
lcsts_part_1=lcsts_part_1.reset_index(drop=True)
lcsts_part_1=pd.concat([lcsts_part_1[1::2].reset_index(drop=True), lcsts_part_1[::2].reset_index(drop=True)], axis=1)
lcsts_part_1.columns=['document', 'summary']
 
lcsts_part_2=pd.read_table('../../resources/corpus/LCSTS_ORIGIN/LCSTS_ORIGIN/DATA/PART_II.txt', header=None,
                           warn_bad_lines=True, error_bad_lines=False, sep='<[/d|/s|do|su|sh][^a].*>', encoding='utf-8')
lcsts_part_2=lcsts_part_2[0].dropna()
lcsts_part_2=lcsts_part_2.reset_index(drop=True)
lcsts_part_2=pd.concat([lcsts_part_2[1::2].reset_index(drop=True), lcsts_part_2[::2].reset_index(drop=True)], axis=1)
lcsts_part_2.columns=['document', 'summary']
 
dataset_train = Dataset.from_dict(lcsts_part_1)
dataset_valid = Dataset.from_dict(lcsts_part_2)
 
TokenModel = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(TokenModel)
 
 
def preprocess_function(examples):
    inputs = [str(doc) for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    inputs = [str(doc) for doc in examples["summary"]]
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(inputs, max_length=max_target_length, truncation=True)
 
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
 
 
tokenized_datasets_t = dataset_train.map(preprocess_function, batched=True)
tokenized_datasets_v = dataset_valid.map(preprocess_function, batched=True)
 
tokenized_datasets = datasets.DatasetDict({"train":tokenized_datasets_t,"validation": tokenized_datasets_v})
print(tokenized_datasets)

# # 这样就可以使用transformer加载了
# trainer = Seq2SeqTrainer(
#     model,
#     args,
#     train_dataset=tokenized_datasets["train"],# 训练集
#     eval_dataset=tokenized_datasets["validation"],# 验证集
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )
 
# train_result = trainer.train()
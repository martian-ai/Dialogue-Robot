# import argparse
# import torch
# import sys
# import torch.nn as nn

# from torch.utils.data import DataLoader

# sys.path.append('..')
# from modules.alpha_learner.nn_with_bert.function.modeling import BertConfig
# from modules.utils.tokenizer.bert import Tokenizer4Bert 
# from modules.utils.dataset.MatchDataset import MatchDatasetSingleLine
# from modules.modules_args import opt

# class MatchPredictService():
#     def __init__(self, opt, best_model_path):
#         self.opt = opt
#         self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.tokenizer_type)
#         # bert = BertModel.from_pretrained(opt.pretrained_bert_name)  # 此处不是最终的model
#         print(opt.config_file)
#         config = BertConfig(vocab_size_or_config_json_file=opt.config_file)
#         #self.model = nn.DataParallel(opt.model_class(config, opt)).to(opt.device)  # 加载设置特定参数的模型
#         self.model = opt.model_class(config, opt).to(opt.device)  # 加载设置特定参数的模型
#         self.model.load_state_dict(torch.load(best_model_path))  # TODO

#     def predict(self, query1, query2):
#         predict_data = MatchDatasetSingleLine(query1, query2, self.tokenizer, self.opt.max_seq_len)
#         test_data_loader = DataLoader(dataset=predict_data, batch_size=self.opt.batch_size, shuffle=False)
#         self.model.eval()
#         with torch.no_grad():
#             for _, t_batch in enumerate(test_data_loader):
#                 t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
#                 t_outputs = self.model(t_inputs)
#         return t_outputs


# if __name__ == "__main__":

#     match_api = MatchPredictService(opt, 'state_dict/bert-match_demo-match_val_acc_0.9')
#     #match_api = MatchPredictService(opt, 'state_dict/bert-match_match-car-2_val_acc_0.7415')
#     print('最终结果', match_api.predict('放其他音乐忘了没有', '我要听音乐'))
#     print('最终结果', match_api.predict('我要听音乐', '我要听音乐'))
#     print('最终结果', match_api.predict('除了车之外还有其他的可以体验嘛', '想说除了车之外还有其他的可以体验嘛'))
#     print('最终结果', match_api.predict('除了车之外还有其他的可以体验嘛', 'xxx'))

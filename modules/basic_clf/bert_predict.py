import sys

import torch
from torch.utils.data import DataLoader

sys.path.append('..')
from modules.alpha_learner.nn_with_bert.function.modeling import \
    BertConfig  # TODO
from modules.modules_args import opt
from modules.utils.dataset.CLFDataset import CLFDatasetSingleLine
from modules.utils.tokenizer.bert import Tokenizer4Bert


class CLFPredictService():
    def __init__(self, opt, best_model_path):
        self.opt = opt
        self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.tokenizer_type)
        # bert = BertModel.from_pretrained(opt.pretrained_bert_name)  # 此处不是最终的model
        config = BertConfig(vocab_size_or_config_json_file=opt.config_file)
        #self.model = nn.DataParallel(opt.model_class(config, opt)).to(opt.device)  # 加载设置特定参数的模型
        self.model = opt.model_class(config, opt).to(opt.device)  # 加载设置特定参数的模型
        self.model.load_state_dict(torch.load(best_model_path))  # TODO

    def predict(self, query):
        predict_data = CLFDatasetSingleLine(query, self.tokenizer)
        train_data_loader = DataLoader(dataset=predict_data, batch_size=self.opt.batch_size, shuffle=True)
        self.model.eval()
        with torch.no_grad():
            for _, t_batch in enumerate(train_data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_outputs = self.model(t_inputs)
        return t_outputs


if __name__ == "__main__":

    clf_api = CLFPredictService(opt, 'state_dict/bert-clf_demo-sentiment-3_val_acc_0.7952')
    print('最终结果', clf_api.predict('定一个三点的闹铃'))
    print('最终结果', clf_api.predict('我要听音乐'))
    print('最终结果', clf_api.predict('打开台灯'))

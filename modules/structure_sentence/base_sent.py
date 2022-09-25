

class BaseSent(object):
    def __init__(self, text) -> None:
        super().__init__()
        self.text = text
        self.seg = []
        self.pos = []
        self.ner = []
        self.dp = []
        self.srl = []
        self.retall = [] # 1. 反向翻译 2. 生成模型
        self.keywords = []
        self.sent_emb_bert = [] # bert cls
        self.sent_emb_topic = [] # topic
        self.res = []
from modules.question_answering.bert_sinlgespan.predict import \
    QA_BERT_SingleSpan
from modules.question_generate.unilm.predict import QG_Unilm_Full

qg_api = QG_Unilm_Full('resources/model/qg_unilm/model.10.bin')


def docQuestionGeneration(doc, answer):
    qg_result = qg_api.predict(doc, answer)  # 输入是 整个文档，不需要提前分句
    return qg_result


qa_api = QA_BERT_SingleSpan('resources/model/qa_bert_singlespan/pytorch_model.bin', 'resources/model/qa_bert_singlespan/config.json')


def docQuestionAnswering(doc, question):
    qa_result = qa_api.predict(doc, question)  # 输入是 整个文档，不需要提前分句
    return qa_result

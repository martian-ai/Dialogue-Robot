
# def pairs_ranking(sentence, questions, answers):
#     """
#     提取featrue 依赖ltp
#     排序模型需要重新训练， ltp 的特征好改变为当前basicNLP 的特征

#     # TODO 改造 question-answer 模型 基于transformers 
#     """
#     all_results = []
#     if len(questions) == 0:
#         all_results = []
#     else:
#         questions_score = get_pairs_rank_score(xgb_model, questions)  # TODO
#         for tmp_score, tmp_question, tmp_answer in zip(questions_score, questions, answers):
#             tmp_sample = OrderedDict()
#             tmp_sample['sentence'] = sentence
#             tmp_sample['question'] = tmp_question
#             tmp_sample['answer'] = tmp_answer
#             tmp_sample['score'] = str(tmp_score)
#             all_results.append(tmp_sample)
#     return all_results
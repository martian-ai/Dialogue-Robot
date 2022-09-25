
# def docRuleQuesitonGeneration():
#     """
#     functions:

#     params:

#     returns:

#     todo:
#         依赖缺失的模块打造

#     """

#     inputDoc = request.args.get("inputDoc")
#     sentence_list = sentence_segment(inputDoc)
#     all_results = []
#     for inputText in sentence_list:
#         inputText = inputText.strip('[.。！？!?]')
#         rule_pairs = get_rule_pairs(inputText, exist_questions=set())
#         # print('rule', rule_pairs)
#         semantic_pairs = get_semantic_pairs(inputText, exist_questions=set())
#         # print('semantic', semantic_pairs)
#         syntax_pairs = get_syntax_pairs(inputText, exist_questions=set())
#         # print('syntax', syntax_pairs)
#         questions = rule_pairs[0] + semantic_pairs[0] + syntax_pairs[0]
#         answers = rule_pairs[1] + semantic_pairs[1] + syntax_pairs[1]
#         if len(questions) == 0:
#             all_results = []
#         else:
#             questions_score = get_pairs_rank_score(xgb_model, questions)
#             for tmp_score, tmp_question, tmp_answer in zip(questions_score, questions, answers):
#                 tmp_sample = OrderedDict()
#                 tmp_sample['sentence'] = inputText
#                 tmp_sample['question'] = tmp_question
#                 tmp_sample['answer'] = tmp_answer
#                 tmp_sample['score'] = str(tmp_score)
#                 all_results.append(tmp_sample)
#     return json.dumps({'inputDoc':inputDoc, 'results': all_results}, ensure_ascii=False)

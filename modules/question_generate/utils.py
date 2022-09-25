import re

def line_cleanser(line):
    clean_line = re.sub('第.*?条', '', line) 
    clean_line = re.sub('^〔.*〕','', clean_line) 
    clean_line = re.sub('^（.*）', '', clean_line) 
    clean_line = re.sub('^[一二三四五六七八九十]*、', '', clean_line) 
    clean_line = clean_line.strip()
    return clean_line

def safe_add(question, a, exist_questions, questions, answers):
    if question not in exist_questions:
        """
        句子中，；：超过2个就不输出
        """
        sep_counter = 0
        for char in question:
            if sep_counter > 2:
                return
            if char in ['，', '；', '：']:
                sep_counter += 1
        
        exist_questions.add(question)
        questions.append(question)
        answers.append(a)
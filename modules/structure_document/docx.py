import re
import docx
from docx import Document

def regular_process_docx(filename):
    doc = docx.Document(filename)
    paras = doc.paragraphs
    body = {'contents': []}
    tmp_content = {'h1':'', 'clean_h1':'', 'h2':'', 'clean_h2':'', 'content': ''}
    h1 = ''
    match_term = ''
    for idx, each_p in enumerate(paras):
        prgph_text = each_p.text
        body['full_text'] = body.get('full_text', '') + prgph_text 
        prgph_text = prgph_text.replace('\u3000\u3000', '')
        prgph_text = prgph_text.replace('    ', '')
        if '系列解读' in prgph_text:
            body['org_title'] = prgph_text
            body['true_title'] = prgph_text.strip().split('：')[1]
            match_term += body['true_title']
            continue
        if idx == 0:
            body['org_title'] = prgph_text
            if ' ' in prgph_text:
                body['true_title'] = prgph_text.strip().split()[1]
            else:
                body['true_title'] = prgph_text
        if '文章来源' in prgph_text:
            body['source'] = prgph_text
            continue
        h1_pattern = re.compile("^[一二三四五六七八九十]+、")
        h1_match = re.search(h1_pattern, prgph_text)
        h2_pattern = re.compile("^（[一二三四五六七八九十]+）")
        h2_match = re.search(h2_pattern, prgph_text)

        
        if h1_match:
            if tmp_content['h1'] or tmp_content['h2'] or tmp_content['content']:
                body['contents'].append(tmp_content)
                tmp_content = {'h1':'', 'clean_h1':'', 'h2':'', 'clean_h2':'', 'content': ''}
            h1 = prgph_text
            clean_h1 = re.sub(h1_pattern, '', h1)
            tmp_content['h1'] = h1
            tmp_content['clean_h1'] = clean_h1
            match_term += clean_h1
            
        elif h2_match:
            if tmp_content['h1'] or tmp_content['h2'] or tmp_content['content']:
                body['contents'].append(tmp_content)
                tmp_content = {'h1':'', 'clean_h1':'', 'h2':'', 'clean_h2':'', 'content': ''}
            tmp_content['h1'] = h1
            tmp_content['clean_h1'] = clean_h1
            tmp_content['h2'] = prgph_text
            tmp_content['clean_h2'] = re.sub(h2_pattern, '', prgph_text)
            match_term += tmp_content['clean_h2']
        else:
            tmp_content['content'] = tmp_content.get('content', '')+prgph_text


    if tmp_content['h1'] or tmp_content['h2'] or tmp_content['content']:
        body['contents'].append(tmp_content)
    body['match_term'] = match_term
    return body
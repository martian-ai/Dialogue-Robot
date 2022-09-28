with open('../../resources/corpus/solutions/mining_qa/document/三体.txt', mode='r', encoding='utf-8') as f:
    lines = f.readlines()[150:180]

print(lines)

import requests

for line in lines:
    if len(line) < 30:
        continue
    response = requests.get('http://127.0.0.1:5061/doc_qa_mining?topK=5&inputDoc='+line)
    #response = requests.get('http://11.70.128.84/doc_mining?topK=5&inputDoc='+line)
    print(response.json())

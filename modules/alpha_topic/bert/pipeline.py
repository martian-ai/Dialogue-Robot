from bertopic import BERTopic
docs = []
with open('../../../resources/corpus/solutions/mining_qa/document/大刘全篇.txt', mode='r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines[:100]:
        docs.append(line.strip())
    topic_model = BERTopic(language='chinese (simplified)', min_topic_size=3, nr_topics='auto')
    topics, probs = topic_model.fit_transform(docs)
    print(len(topics))
    print(len(probs))
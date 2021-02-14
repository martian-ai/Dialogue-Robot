# pipeline
## 提取query中信息
## 记录历史信息
## 根据query中信息和历史信息来 为 nlg 提供支持

"""
state_detection
"""

class state_detection(object):

    """
    多轮领域检测
    """
    def multi_domain_detect(self):
        pass

    """
    多轮意图检测
    """
    def multi_intent_detect(self):
        pass

    """
    多轮意图检测
    """
    def multi_sentiment_detect(self):
        pass

    """
    时间/地点/人物
    """





"""
Context Query Understanding
"""
class context_query_understanding(object):
    
    """
    实体链接
    """
    def entity_link(self, input_str):
        pass

    """
    指代消解
    """
    def anaphora_resolution(self, input_str):
        pass

    """
    句子补全 
    """
    def sentence_completion(self,input_str):
        pass

"""
user simulation
"""
class user_simulation(object):
    pass


"""
agent setting
"""
class agent_setting(object):
    pass


"""
dialogue policy
"""
class dialogue_policy(object):
    pass



if __name__ == '__main__':
    pass
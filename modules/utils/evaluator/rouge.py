from rouge import Rouge
def rouge(a,b):
    rouge = Rouge()  
    rouge_score = rouge.get_scores(a,b, avg=True) # a和b里面包含多个句子的时候用
    print(rouge_score)
    rouge_score1 = rouge.get_scores(a,b)  # a和b里面只包含一个句子的时候用
    # 以上两句可根据自己的需求来进行选择
    r1 = rouge_score["rouge-1"]
    r2 = rouge_score["rouge-2"]
    rl =rouge_score["rouge-l"]
    
    return rl
    #return r1, r2, rl

def main():
    a = ["i am a student from china", "the cat was found under the bed"]  # 预测摘要
    b = ["i am student from school on japan", "the cat was under the bed"]  # 参考摘要
    rl = rouge(a,b)
    #print(r1)
    #print(r2)
    print(rl)

if __name__ == '__main__':
    main()

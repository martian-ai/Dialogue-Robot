# Fairy tale basecase 

>>> section_idx 0, section key alleleiraugh-or-the-many-furred-creature-story--1 >>> 
>>>>>>  ground answer : golden hair >><< ground question : what kind of hair did the wife have <<<<<<
>>> top 3 socre is [0.5454545454545454, 0.47619047619047616, 0.4444444444444445] 
>>> top 1 text : such golden hair  what did the queen want from a second wife
>>> top 2 text : golden hair  what did the queen want from her second husband
>>> top 3 text : no one queen  what did the wife say

top2 question 争取， top 2 的 answer 有问题



>>>>>>  ground answer : the sun, moon, and star dresses >><< ground question : what did the princess put in a nut-shell <<<<<<
>>> top 3 socre is [0.4615384615384615, 0.4444444444444445, 0.43478260869565216] 
>>> top 1 text : star dresses  what did the youngest princess make from her treasures
>>> top 2 text : a nut-shell  what did the youngest princess put in her dress
>>> top 3 text : the three dresses  what did the king order

top1 和 top3 选择的answer 没问题，语义上一致，top1 是 答案的一部分
question 部分 都没有出现 put in nut-shell

>>>>>>  ground answer : she did not want the king to find her >><< ground question : why did the princess not tell the huntsmen she was the king's daughter <<<<<<
>>> top 3 socre is [0.36363636363636365, 0.30303030303030304, 0.29411764705882354] 
>>> top 1 text : the king  why did the huntsmen take many-furred creature
>>> top 2 text : king's  why did the huntsmen take many-furred creature
>>> top 3 text : the sun rose  why did the huntsmen take many-furred creature

完全错误，答案会问题的角度完全不对
产生的answer 过短

>>>>>>  ground answer : she begged hard >><< ground question : why did the cook let the many-furred creature go up for the usual time <<<<<<
>>> top 3 socre is [0.8823529411764706, 0.45161290322580644, 0.45161290322580644] 
>>> top 1 text : she begged hard  why did the cook let many-furred creature go up forthe usual time
>>> top 2 text : third  why did the cook think that many-furred creature was a witch
>>> top 3 text : two  why did the cook think that many-furred creature was a witch

top 1 答案质量很高


>>>>>>  ground answer : withered and browned >><< ground question : what happened to the roses when the wife put the wreath on her daughter's head <<<<<<
>>> top 3 socre is [0.3888888888888889, 0.3448275862068966, 0.3225806451612903] 
>>> top 1 text : she pretended to despise the wreath  why did the stepmother want a wreath like her daughter's
>>> top 2 text : the roses  what did the stepmother do to her daughter
>>> top 3 text : the birds  what did the stepmother do to her daughter's wreath
>
答案抽取的质量都不理想
考虑答案抽取替换成生成模型，并在生成时 融合 词性 信息
# Seq2Seq

## 原始方法
+ Encoder-Decoder 结构
    + 整体结构
    ![20220322153259](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220322153259.png)
    + decoder 处理方式1
    ![20220322153614](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220322153614.png)
    + decoder 处理方式2
    ![20220322153627](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220322153627.png)
+ 缺点
    + 使用最后一个位置的隐层输出作为整体信息编码，定长编码存在信息瓶颈
    + 长度越长，前面输入进RNN的信息就越容易被稀释


## 带有Attention
+ 改进
    + Decoder 当前 的 Hidden state 和 全体 Encoder 的 Hidden state 计算Attention权重
        + 权重的计算方式
            + 符号定义
                + EO : Encoder 各个位置的输出
                + H : Decoder 某个位置的隐藏状态
                + FC : 全链接层
                + X : decoder 某个位置的输入
            + Bahdanau 注意力
                + score = FC(tanh(FC(EO) + FC(H)))
            + luong 注意力
                + score = EO * W * H
    + 权重进行softmax
        + attention_weights = softmax(score, axis=1)
    + 使用Attention权重 和 全体 Encoder 的 Hidden state 进行加权求和
        + context = sum(attention_weights * EO , axis = 1)
    + 将求和结果与 decoder 当前位置的 Hidden State 进行拼接
        + final_input = concate(x, context)
    + 之后进行softmax 预测 decoder 当前位置的结果
![20220321162658](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220321162658.png)

+ 缺点
    + encoder和decoder采用lstm，不能并行，且存在一定程度梯度消失和梯度弥散的可能

## Reference
+ 见 Plan.md / Modules / generator / seq2seq


ffn


tokenizer

token_type_ids


output_all_encoded_layers
在forward函数中，如果output_all_encoded_layers=True，那么encoded_layer就是12层transformer的结果，否则只返回最后一层transformer的结果，pooled_output的输出就是最后一层Transformer经过self.pooler层后得到的结果


# Ref
+ https://www.jianshu.com/p/4e139a3260fd
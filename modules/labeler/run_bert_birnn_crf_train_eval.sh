dataset_name='frying-pan'
type_name='entity'
gpu='3'
epoch=10
max_seq_len=512
max_seq_len_predict=1024
learning_rate=9e-6
hidden_layer=6
target_folder=./outputs/${dataset_name}_${type_name}_epoch_${epoch}_hidden_layer_${hidden_layer}_max_seq_len_${max_seq_len}_gpu_${gpu} 
train_flag=True # whether to train model on trainset
eval_flag=False # whether to eval trained model on devset, default is False 
test_flag=True # whether to eval trained model on devset, default is False 
predict_flag=True # whether to predict result on testset by trained model
metric_flag=True # whether run eval.py to calculate metric

if [ "$type_name" == 'emotion' ] ;then
label_list="O,[CLS],[SEP],B-positive,I-positive,B-negative,I-negative,B-moderate,I-moderate"
echo $label_list
fi

if [ "$type_name" == 'entity' ] ;then
label_list="O,[CLS],[SEP],B-3,I-3"
echo $label_list
fi

echo ${target_folder}

if [ "$train_flag" == True  -a  -d "$target_folder" ] ;then
/bin/rm -rf $target_folder
mkdir $target_folder
elif [ "$train_flag" == True -a  ! -d "$target_folder" ];then
mkdir $target_folder
fi

if [ $train_flag == True -o $eval_flag == True -o $test_flag == True ] ;then
# Train or Predict
python models/BERT_BIRNN_CRF.py \
    --task_name="NER"  \
    --type_name=${type_name} \
    --label_list=${label_list} \
    --gpu=${gpu} \
    --do_lower_case=False \
    --do_train=${train_flag}   \
    --do_eval=${eval_flag}   \
    --do_test=${test_flag}   \
    --do_predict=${predict_flag} \
    --data_dir=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/${dataset_name}/slot \
    --vocab_file=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/resources/chinese_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/resources/chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/resources/chinese_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=$max_seq_len   \
    --train_batch_size=16   \
    --learning_rate=${learning_rate}   \
    --num_train_epochs=$epoch   \
    --output_dir=$target_folder
fi

if [ $metric_flag == True ] ;then
# delete lines which contain [CLS], [SEP]  
cp ${target_folder}/${type_name}_test_results.txt ${target_folder}/${type_name}_test_results.txt-backup
sed -i '/SEP/d' ${target_folder}/${type_name}_test_results.txt
sed -i '/CLS/d' ${target_folder}/${type_name}_test_results.txt

python evals/evaluate.py \
    --ground_text_path=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/${dataset_name}/slot/test.txt \
    --predict_label_path=${target_folder}/${type_name}_test_results.txt 
fi

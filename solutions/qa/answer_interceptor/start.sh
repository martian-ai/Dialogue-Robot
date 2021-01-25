CURRENT_DIR=`pwd`
export DATA_DIR=$CURRENT_DIR/data

nohup python bert_qa.py \
  --model_type bert \
  --model_name_or_path bert-base-chinese \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $DATA_DIR/cmrc2018_train.json \
  --predict_file $DATA_DIR/cmrc2018_dev.json \
  --per_gpu_train_batch_size 24 \
  --learning_rate 3e-5 \
  --num_train_epochs 200.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --output_dir output > cmrc_bert_qa.log &

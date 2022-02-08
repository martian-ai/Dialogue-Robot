CURRENT_DIR=`pwd`
export DATA_DIR=$CURRENT_DIR/data

python -B ../../modules/interceptor/interactive.py \
  --model_type bert \
  --model_name_or_path bert-base-chinese \
  --do_eval \
  --do_lower_case \
  --per_gpu_train_batch_size 24 \
  --max_seq_length 384 \
  --state_dict '../../resources/models/mrc/pytorch_model.bin' \
  --doc_stride 128

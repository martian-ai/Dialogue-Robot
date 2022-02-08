EPOCH=20
DATA_DIR=..//dataset/merge_baidu_raw_cmrc2018_full_context/train
# resources/dataset/passage-answer-question/merge_baidu_cmrc/train
OUTPUT_DIR=./output/merge_baidu_cmrc_${EPOCH}/
mv ${OUTPUT_DIR} ${OUTPUT_DIR}-backup
rm -rf ${OUTPUT_DIR}
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python Run.py \
  --do_train \
  --num_workers 0 \
  --bert_model bert-base-chinese \
  --new_segment_ids \
  --tokenized_input \
  --data_dir ${DATA_DIR} \
  --src_file train_pa_list.txt \
  --tgt_file train_q_list.txt \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --max_seq_length 512 \
  --max_position_embeddings 512 \
  --mask_prob 0.7 \
  --max_pred 48 \
  --train_batch_size 128 \
  --gradient_accumulation_steps 2 \
  --learning_rate 0.00002 \
  --warmup_proportion 0.1 \
  --label_smoothing 0.1 \
  --num_train_epochs ${EPOCH} > logs/full_context_finetune.log &

tail -f logs/full_context_finetune.log

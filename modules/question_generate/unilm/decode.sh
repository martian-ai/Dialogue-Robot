EPOCH=20
DATA_DIR=./dataset/merge_baidu_raw_cmrc2018_full_context/test
MODEL_RECOVER_PATH=./output/merge_baidu_cmrc_${EPOCH}/bert_save/model.20.bin
EVAL_SPLIT=test
#export PYTORCH_PRETRAINED_BERT_CACHE=./models/bert-cased-pretrained-cache
# run decoding
python3 biunilm/decode_seq2seq.py \
  --bert_model bert-base-chinese \
  --new_segment_ids --mode s2s \
  --input_file ${DATA_DIR}/test_pa_list.txt \
  --split ${EVAL_SPLIT} --tokenized_input \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 512 \
  --max_tgt_length 48 \
  --batch_size 16 \
  --beam_size 1 \
  --length_penalty 0


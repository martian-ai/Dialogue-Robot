rm -rf input/train_files/train.txt
ps aux|grep train.py|grep -v grep|cut -c 9-15|xargs kill -9
nohup python train.py  -device 3 -batch_size 128 -model CNN_RNN -seed 1 -max_epoch 20 -output output/ > sum.log &
tail -f sum.log

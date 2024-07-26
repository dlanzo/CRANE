# !/bin/bash

python3 train.py \
	--device  'cuda' \
	--padding 'circular' \
	--size 32 \
	--seed 10 \
	--epochs 1000\
	--lr 1e-4 \
	--batch 2 \
	--dual \
	--rotation90 \
	--reflection \
	--train_set 'data/train_set.txt' \
	--valid_set 'data/valid_set.txt' \
	--id 'TRAINING' \
	--subseq_min 1 \
	--subseq_max 50 \
	--logfreq 1 \
	--kernel_size 3 \
	--hidden 2 \
	--channels 10 \
	--nproc 4 \
	--ramp \
	--ramp_length 25 \
	--start_ramp 1 \
	--threeD \
	--bias \
	--divergence \

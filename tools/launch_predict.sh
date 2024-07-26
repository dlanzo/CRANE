# !/bin/bash

python3 -u run_3D.py \
	--device 'cuda' \
	--size 64 \
	--kernel_size 3 \
	--hidden 2 \
	--channels 10 \
	--padding 'circular' \
	--seed 1 \
	--id 'compare_longevo' \
	--in_frames 1 \
	--tot_frames 100 \
	--model_name 'models/model.pt' \
	--load_image 'in/input.npy' \
	--nproc 0 \
	--threeD \
	--bias \
	--divergence \
	--save_every 20

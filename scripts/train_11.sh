export CUDA_VISIBLE_DEVICES="0"
export NGPU=1
export WORK_DIR=./output/sana600_no
export LOAD_FROM=''

torchrun --nproc_per_node=$NGPU --master_port=21541 scripts/train.py --config_path $1 --model.load_from $LOAD_FROM --work_dir $WORK_DIR --train.train_batch_size 2 --model.use_pe true
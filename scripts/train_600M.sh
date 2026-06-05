export CUDA_VISIBLE_DEVICES="0"
export NGPU=1
export WORK_DIR=./output/0815
export LOAD_FROM=hf://Efficient-Large-Model/Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth

torchrun --nproc_per_node=$NGPU --master_port=21540 scripts/train.py --config_path $1 --model.load_from $LOAD_FROM --work_dir $WORK_DIR --resume_from latest --resume_from_sana true --train.train_batch_size 8 --model.use_pe true
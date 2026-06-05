export CUDA_VISIBLE_DEVICES="0"
export NGPU=1
export WORK_DIR=./output/loss_24
export LOAD_FROM=hf://Efficient-Large-Model/Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth

torchrun --nproc_per_node=$NGPU --master_port=21541 scripts/train_loss.py --config_path $1 --model.load_from $LOAD_FROM --work_dir $WORK_DIR --resume_from latest --train.train_batch_size 2 --resume_from_sana false --model.use_pe true
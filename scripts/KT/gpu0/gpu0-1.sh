# CMNIST vanilla 1pct

python train.py \
    --base \
    --train_vanilla \
    --dataset=cmnist \
    --percent=1pct \
    --lr=0.01 \
    --data_dir '/home/work/human_pose/Debiasing/benchmarks' \
    --preproc_dir '/home/work/human_pose/Debiasing/preproc' \
    --projcode 'CMNIST 1pct' \
    --run_name 'vanilla-base' \
    --device 'cuda' \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 0 \
    --wandb
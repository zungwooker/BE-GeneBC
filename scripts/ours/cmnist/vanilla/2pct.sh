python train.py \
    --train_vanilla_ours \
    --dataset=cmnist \
    --percent=2pct \
    --lr=0.01 \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --projcode 'CMNIST 2pct' \
    --run_name 'vanilla-ours' \
    --device 'cuda' \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 0 \
    --wandb

python train.py \
    --train_vanilla_ours \
    --dataset=cmnist \
    --percent=2pct \
    --lr=0.01 \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --projcode 'CMNIST 2pct' \
    --run_name 'vanilla-ours' \
    --device 'cuda' \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 42 \
    --wandb
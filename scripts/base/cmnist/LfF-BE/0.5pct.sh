python train.py \
    --train_lff_be_base \
    --dataset=cmnist \
    --percent=0.5pct \
    --lr=0.01 \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --projcode 'CMNIST 0.5pct' \
    --run_name 'LfF-BE-base' \
    --device 'cuda' \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 0 \
    --wandb

python train.py \
    --train_lff_be_base \
    --dataset=cmnist \
    --percent=0.5pct \
    --lr=0.01 \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --projcode 'CMNIST 0.5pct' \
    --run_name 'LfF-BE-base' \
    --device 'cuda' \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 42 \
    --wandb
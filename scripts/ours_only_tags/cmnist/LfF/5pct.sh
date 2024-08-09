# All to D는 train_g_dataset에 원본 데이터 전체만 들어감

python train.py \
    --ours \
    --only_tags \
    --only_no_tags \
    --train_lff \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --projcode 'CMNIST 5pct' \
    --run_name 'LfF-ours-only-tags-all-to-D' \
    --device 'cuda' \
    --gpu_num 2 \
    --fix_randomseed \
    --seed 0 \
    --wandb

python train.py \
    --ours \
    --only_tags \
    --only_no_tags \
    --train_lff \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --projcode 'CMNIST 5pct' \
    --run_name 'LfF-ours-only-tags-all-to-D' \
    --device 'cuda' \
    --gpu_num 2 \
    --fix_randomseed \
    --seed 42 \
    --wandb
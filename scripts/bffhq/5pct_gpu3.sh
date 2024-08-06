# BE + LFF + Ours
# All -> B.Net
# All -> D.Net
# 5pct
# Seed: 42, 43, 44

python train.py \
    --gpu_num 3 \
    --fix_randomseed \
    --seed 42 \
    --train_lff_be_ours_all \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=bffhq \
    --percent=5pct \
    --lr=0.0001 \
    --exp=lff_be_ours_bffhq_5 \
    --num_steps 50000 \
    --projcode 'BFFHQ BE + LFF + OURS' \
    --run_name '5pct A->B, A->D' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 3 \
    --fix_randomseed \
    --seed 43 \
    --train_lff_be_ours_all \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=bffhq \
    --percent=5pct \
    --lr=0.0001 \
    --exp=lff_be_ours_bffhq_5 \
    --num_steps 50000 \
    --projcode 'BFFHQ BE + LFF + OURS' \
    --run_name '5pct A->B, A->D' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 3 \
    --fix_randomseed \
    --seed 44 \
    --train_lff_be_ours_all \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=bffhq \
    --percent=5pct \
    --lr=0.0001 \
    --exp=lff_be_ours_bffhq_5 \
    --num_steps 50000 \
    --projcode 'BFFHQ BE + LFF + OURS' \
    --run_name '5pct A->B, A->D' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb
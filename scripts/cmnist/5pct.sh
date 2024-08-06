# LFF + Ours
# Origin -> B.Net
# All -> D.Net
# 5pct
# Seed: 42, 43, 44

python train.py \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 42 \
    --train_lff_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_ours_cmnist_5 \
    --num_steps 50000 \
    --projcode 'CMNIST LFF + OURS' \
    --run_name '5pct O->B, A->D' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 43 \
    --train_lff_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_ours_cmnist_5 \
    --num_steps 50000 \
    --projcode 'CMNIST LFF + OURS' \
    --run_name '5pct O->B, A->D' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 44 \
    --train_lff_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_ours_cmnist_5 \
    --num_steps 50000 \
    --projcode 'CMNIST LFF + OURS' \
    --run_name '5pct O->B, A->D' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

# LFF + Ours
# All -> B.Net
# All -> D.Net
# 5pct
# Seed: 42, 43, 44

python train.py \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 42 \
    --train_lff_ours_all \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_ours_cmnist_5 \
    --num_steps 50000 \
    --projcode 'CMNIST LFF + OURS' \
    --run_name '5pct A->B, A->D' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 43 \
    --train_lff_ours_all \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_ours_cmnist_5 \
    --num_steps 50000 \
    --projcode 'CMNIST LFF + OURS' \
    --run_name '5pct A->B, A->D' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 44 \
    --train_lff_ours_all \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_ours_cmnist_5 \
    --num_steps 50000 \
    --projcode 'CMNIST LFF + OURS' \
    --run_name '5pct A->B, A->D' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

# BE + LFF + Ours
# Origin -> B.Net
# All -> D.Net
# 5pct
# Seed: 42, 43, 44

python train.py \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 42 \
    --train_lff_be_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_ours_cmnist_5 \
    --num_steps 50000 \
    --projcode 'CMNIST BE + LFF + OURS' \
    --run_name '5pct O->B, A->D' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 43 \
    --train_lff_be_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_ours_cmnist_5 \
    --num_steps 50000 \
    --projcode 'CMNIST BE + LFF + OURS' \
    --run_name '5pct O->B, A->D' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 44 \
    --train_lff_be_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_ours_cmnist_5 \
    --num_steps 50000 \
    --projcode 'CMNIST BE + LFF + OURS' \
    --run_name '5pct O->B, A->D' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

# BE + LFF + Ours
# All -> B.Net
# All -> D.Net
# 5pct
# Seed: 42, 43, 44

python train.py \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 42 \
    --train_lff_be_ours_all \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_ours_cmnist_5 \
    --num_steps 50000 \
    --projcode 'CMNIST BE + LFF + OURS' \
    --run_name '5pct A->B, A->D' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 43 \
    --train_lff_be_ours_all \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_ours_cmnist_5 \
    --num_steps 50000 \
    --projcode 'CMNIST BE + LFF + OURS' \
    --run_name '5pct A->B, A->D' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 44 \
    --train_lff_be_ours_all \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_ours_cmnist_5 \
    --num_steps 50000 \
    --projcode 'CMNIST BE + LFF + OURS' \
    --run_name '5pct A->B, A->D' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb
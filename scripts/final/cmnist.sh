# Done.
# python train.py \
#     --ours \
#     --half_generated \
#     --train_lff \
#     --dataset=cmnist \
#     --percent=0.5pct \
#     --lr=0.01 \
#     --data_dir '/mnt/sdd/Debiasing/benchmarks' \
#     --preproc_dir '/mnt/sdd/Debiasing/preproc' \
#     --projcode 'BiasEdit CMNIST 0.5pct' \
#     --run_name 'LfF-half-generated' \
#     --device 'cuda' \
#     --gpu_num 1 \
#     --fix_randomseed \
#     --seed 0 \
#     --wandb

python train.py \
    --ours \
    --half_generated \
    --train_lff \
    --dataset=cmnist \
    --percent=1pct \
    --lr=0.01 \
    --data_dir '/mnt/sdd/Debiasing/benchmarks' \
    --preproc_dir '/mnt/sdd/Debiasing/preproc' \
    --projcode 'BiasEdit CMNIST 1pct' \
    --run_name 'LfF-half-generated' \
    --device 'cuda' \
    --gpu_num 1 \
    --fix_randomseed \
    --seed 0 \
    --wandb

python train.py \
    --ours \
    --half_generated \
    --train_lff \
    --dataset=cmnist \
    --percent=2pct \
    --lr=0.01 \
    --data_dir '/mnt/sdd/Debiasing/benchmarks' \
    --preproc_dir '/mnt/sdd/Debiasing/preproc' \
    --projcode 'BiasEdit CMNIST 2pct' \
    --run_name 'LfF-half-generated' \
    --device 'cuda' \
    --gpu_num 1 \
    --fix_randomseed \
    --seed 0 \
    --wandb

python train.py \
    --ours \
    --half_generated \
    --train_lff \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --data_dir '/mnt/sdd/Debiasing/benchmarks' \
    --preproc_dir '/mnt/sdd/Debiasing/preproc' \
    --projcode 'BiasEdit CMNIST 5pct' \
    --run_name 'LfF-half-generated' \
    --device 'cuda' \
    --gpu_num 1 \
    --fix_randomseed \
    --seed 0 \
    --wandb
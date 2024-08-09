# Done.
python train.py \
    --ours \
    --half_generated \
    --train_lff \
    --dataset=cifar10c \
    --percent=0.5pct \
    --lr=0.0005 \
    --data_dir '/mnt/sdd/Debiasing/benchmarks' \
    --preproc_dir '/mnt/sdd/Debiasing/preproc' \
    --projcode 'BiasEdit CIFAR10C 0.5pct' \
    --run_name 'LfF-half-generated' \
    --device 'cuda' \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 0 \
    --wandb

python train.py \
    --ours \
    --half_generated \
    --train_lff \
    --dataset=cifar10c \
    --percent=1pct \
    --lr=0.001 \
    --data_dir '/mnt/sdd/Debiasing/benchmarks' \
    --preproc_dir '/mnt/sdd/Debiasing/preproc' \
    --projcode 'BiasEdit CIFAR10C 1pct' \
    --run_name 'LfF-half-generated' \
    --device 'cuda' \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 0 \
    --wandb

python train.py \
    --ours \
    --half_generated \
    --train_lff \
    --dataset=cifar10c \
    --percent=2pct \
    --lr=0.001 \
    --data_dir '/mnt/sdd/Debiasing/benchmarks' \
    --preproc_dir '/mnt/sdd/Debiasing/preproc' \
    --projcode 'BiasEdit CIFAR10C 2pct' \
    --run_name 'LfF-half-generated' \
    --device 'cuda' \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 0 \
    --wandb

python train.py \
    --ours \
    --half_generated \
    --train_lff \
    --dataset=cifar10c \
    --percent=5pct \
    --lr=0.001 \
    --data_dir '/mnt/sdd/Debiasing/benchmarks' \
    --preproc_dir '/mnt/sdd/Debiasing/preproc' \
    --projcode 'BiasEdit CIFAR10C 5pct' \
    --run_name 'LfF-half-generated' \
    --device 'cuda' \
    --gpu_num 0 \
    --fix_randomseed \
    --seed 0 \
    --wandb
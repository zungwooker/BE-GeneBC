# Running
python train.py \
    --ours \
    --half_generated \
    --train_lff \
    --dataset=bffhq \
    --percent=0.5pct \
    --lr=0.0001 \
    --data_dir '/mnt/sdd/Debiasing/benchmarks' \
    --preproc_dir '/mnt/sdd/Debiasing/preproc' \
    --projcode 'BiasEdit BFFHQ 0.5pct' \
    --run_name 'LfF-half-generated' \
    --device 'cuda' \
    --gpu_num 3 \
    --fix_randomseed \
    --seed 0 \
    --wandb

python train.py \
    --ours \
    --half_generated \
    --train_lff \
    --dataset=bffhq \
    --percent=1pct \
    --lr=0.0001 \
    --data_dir '/mnt/sdd/Debiasing/benchmarks' \
    --preproc_dir '/mnt/sdd/Debiasing/preproc' \
    --projcode 'BiasEdit BFFHQ 1pct' \
    --run_name 'LfF-half-generated' \
    --device 'cuda' \
    --gpu_num 3 \
    --fix_randomseed \
    --seed 0 \
    --wandb

python train.py \
    --ours \
    --half_generated \
    --train_lff \
    --dataset=bffhq \
    --percent=2pct \
    --lr=0.0001 \
    --data_dir '/mnt/sdd/Debiasing/benchmarks' \
    --preproc_dir '/mnt/sdd/Debiasing/preproc' \
    --projcode 'BiasEdit BFFHQ 2pct' \
    --run_name 'LfF-half-generated' \
    --device 'cuda' \
    --gpu_num 3 \
    --fix_randomseed \
    --seed 0 \
    --wandb

python train.py \
    --ours \
    --half_generated \
    --train_lff \
    --dataset=bffhq \
    --percent=5pct \
    --lr=0.0001 \
    --data_dir '/mnt/sdd/Debiasing/benchmarks' \
    --preproc_dir '/mnt/sdd/Debiasing/preproc' \
    --projcode 'BiasEdit BFFHQ 5pct' \
    --run_name 'LfF-half-generated' \
    --device 'cuda' \
    --gpu_num 3 \
    --fix_randomseed \
    --seed 0 \
    --wandb
# Running
python train.py \
    --ours \
    --half_generated \
    --train_lff \
    --dataset=dogs_and_cats \
    --percent=1pct \
    --lr=0.0001 \
    --data_dir '/mnt/sdd/Debiasing/benchmarks' \
    --preproc_dir '/mnt/sdd/Debiasing/preproc' \
    --projcode 'BiasEdit DNC 1pct' \
    --run_name 'LfF-half-generated' \
    --device 'cuda' \
    --gpu_num 6 \
    --fix_randomseed \
    --seed 0 \
    --wandb

python train.py \
    --ours \
    --half_generated \
    --train_lff \
    --dataset=dogs_and_cats \
    --percent=5pct \
    --lr=0.0001 \
    --data_dir '/mnt/sdd/Debiasing/benchmarks' \
    --preproc_dir '/mnt/sdd/Debiasing/preproc' \
    --projcode 'BiasEdit DNC 5pct' \
    --run_name 'LfF-half-generated' \
    --device 'cuda' \
    --gpu_num 6 \
    --fix_randomseed \
    --seed 0 \
    --wandb

python train.py \
    --ours \
    --half_generated \
    --train_lff \
    --dataset=bar \
    --percent=1pct \
    --lr=0.00001 \
    --resnet_pretrained \
    --data_dir '/mnt/sdd/Debiasing/benchmarks' \
    --preproc_dir '/mnt/sdd/Debiasing/preproc' \
    --projcode 'BiasEdit BAR 1pct' \
    --run_name 'LfF-half-generated' \
    --device 'cuda' \
    --gpu_num 6 \
    --fix_randomseed \
    --seed 0 \
    --wandb

python train.py \
    --ours \
    --half_generated \
    --train_lff \
    --dataset=bar \
    --percent=5pct \
    --lr=0.00001 \
    --resnet_pretrained \
    --data_dir '/mnt/sdd/Debiasing/benchmarks' \
    --preproc_dir '/mnt/sdd/Debiasing/preproc' \
    --projcode 'BiasEdit BAR 5pct' \
    --run_name 'LfF-half-generated' \
    --device 'cuda' \
    --gpu_num 6 \
    --fix_randomseed \
    --seed 0 \
    --wandb
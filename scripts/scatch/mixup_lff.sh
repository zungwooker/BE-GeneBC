python train.py \
    --gpu_num 1 \
    --train_lff_ours_mixup \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_5 \
    --projcode 'CMNIST-LFF-OURS-MIXUP' \
    --run_name '5pct' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --fix_randomseed \
    --seed 0 \
    --mixup \
    --wandb

python train.py \
    --gpu_num 1 \
    --train_lff_ours_mixup \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_5 \
    --projcode 'CMNIST-LFF-OURS-MIXUP' \
    --run_name '5pct' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --fix_randomseed \
    --seed 42 \
    --mixup \
    --wandb

python train.py \
    --gpu_num 1 \
    --train_lff_ours_mixup \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=0.5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_5 \
    --projcode 'CMNIST-LFF-OURS-MIXUP' \
    --run_name '0.5pct' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --fix_randomseed \
    --seed 0 \
    --mixup \
    --wandb

python train.py \
    --gpu_num 1 \
    --train_lff_ours_mixup \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=0.5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_5 \
    --projcode 'CMNIST-LFF-OURS-MIXUP' \
    --run_name '0.5pct' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --fix_randomseed \
    --seed 42 \
    --mixup \
    --wandb
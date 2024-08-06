### LfF

# python train.py \
#     --gpu_num 1 \
#     --train_lff_be_ours \
#     --num_bias_models 3 \
#     --agreement 2 \
#     --data_dir '/home/zungwooker/Debiasing/benchmarks' \
#     --device 'cuda' \
#     --dataset=cmnist \
#     --percent=0.5pct \
#     --lr=0.01 \
#     --exp=lff_be_gene_cmnist_0.5 \
#     --num_steps 50000 \
#     --projcode 'LFF-OURS-CMNIST' \
#     --run_name '0.5pct Origin+Gene, B.Net & BCDs get only original dataset' \
#     --preproc_dir '/home/zungwooker/Debiasing/preproc' \
#     --ema \
#     --fix_randomseed \
#     --seed 0

python train.py \
    --gpu_num 1 \
    --train_lff_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=0.5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_0.5 \
    --num_steps 50000 \
    --projcode 'LFF-OURS-CMNIST' \
    --run_name '0.5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --fix_randomseed \
    --seed 0
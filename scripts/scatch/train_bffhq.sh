### LfF + BE

python train.py \
    --gpu_num 1 \
    --train_lff_be_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=bffhq \
    --percent=2pct \
    --lr=0.0001 \
    --exp=lff_be_gene_bffhq_2 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-BFFHQ' \
    --run_name '2pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

# 5pct
python train.py \
    --gpu_num 1 \
    --train_lff_be_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=bffhq \
    --percent=5pct \
    --lr=0.0001 \
    --exp=lff_be_gene_bffhq_5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-BFFHQ' \
    --run_name '5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 1 \
    --train_lff_be_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=bffhq \
    --percent=5pct \
    --lr=0.0001 \
    --exp=lff_be_gene_bffhq_5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-BFFHQ' \
    --run_name '5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 1 \
    --train_lff_be_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=bffhq \
    --percent=5pct \
    --lr=0.0001 \
    --exp=lff_be_gene_bffhq_5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-BFFHQ' \
    --run_name '5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb
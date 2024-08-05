### LfF + BE

python train.py \
    --gpu_num 1 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=bffhq \
    --percent=0.5pct \
    --lr=0.0001 \
    --exp=lff_be_gene_bffhq_0.5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-BFFHQ' \
    --run_name '0.5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 1: BFFHQ 0.5pct training done.\n bffhq 1pct training...'

python train.py \
    --gpu_num 1 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=bffhq \
    --percent=1pct \
    --lr=0.0001 \
    --exp=lff_be_gene_bffhq_1 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-BFFHQ' \
    --run_name '1pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 1: BFFHQ 1pct training done.\n bffhq 2pct training...'

python train.py \
    --gpu_num 1 \
    --train_lff_be \
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
    --wandb \
    --email 'KM 1: BFFHQ 2pct training done.\n bffhq 5pct training...'

python train.py \
    --gpu_num 1 \
    --train_lff_be \
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
    --wandb \
    --email 'KM 1: BFFHQ 5pct training done.\n bffhq BE+Ours Done!'
### LfF + BE

python train.py \
    --gpu_num 3 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=bar \
    --percent=1pct \
    --lr=0.00001 \
    --resnet_pretrained \
    --exp=lff_be_gene_bar_1 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-BAR' \
    --run_name '1pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 3: bar 1pct training done.\n bar 2pct training...'

python train.py \
    --gpu_num 3 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=bar \
    --percent=5pct \
    --lr=0.00001 \
    --resnet_pretrained \
    --exp=lff_be_gene_bar_5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-BAR' \
    --run_name '5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 3: bar 5pct training done.\n bar BE+Ours Done!'
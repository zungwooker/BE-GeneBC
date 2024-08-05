### LfF + BE

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=dogs_and_cats \
    --percent=1pct \
    --lr=0.0001 \
    --exp=lff_be_gene_dogs_and_cats_1 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-DNC' \
    --run_name '1pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 2: dogs_and_cats 1pct training done.\n dogs_and_cats 2pct training...'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=dogs_and_cats \
    --percent=5pct \
    --lr=0.0001 \
    --exp=lff_be_gene_dogs_and_cats_5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-DNC' \
    --run_name '5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 2: dogs_and_cats 5pct training done.\n dogs_and_cats BE+Ours Done!'
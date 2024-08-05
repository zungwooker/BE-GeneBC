### LfF + BE

for i in {1..5}
do
python train.py \
    --gpu_num 0 \
    --train_disent_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=0.5pct \
    --lr=0.01 \
    --exp=disent_be_gene_cmnist_0.5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --curr_step 10000 \
    --lambda_swap 1 \
    --lambda_dis_align 10 \
    --lambda_swap_align 10 \
    --use_lr_decay \
    --lr_decay_step 10000 \
    --lr_gamma 0.5 \
    --projcode 'BE-DISENT-CMNIST' \
    --run_name '0.5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb
done

# python train.py \
#     --gpu_num 0 \
#     --train_disent_be \
#     --data_dir '/home/zungwooker/Debiasing/benchmarks' \
#     --device 'cuda' \
#     --dataset=cmnist \
#     --percent=1pct \
#     --lr=0.01 \
#     --exp=disent_be_gene_cmnist_1 \
#     --num_steps 50000 \
#     --num_bias_models 5 \
#     --agreement 3 \
#     --curr_step 10000 \
#     --lambda_swap 1 \
#     --lambda_dis_align 10 \
#     --lambda_swap_align 10 \
#     --use_lr_decay \
#     --lr_decay_step 10000 \
#     --lr_gamma 0.5 \
#     --projcode 'BE-DISENT-CMNIST' \
#     --run_name '1pct Origin+Gene, B.Net & BCDs get only original dataset' \
#     --preproc_dir '/home/zungwooker/Debiasing/preproc' \
#     --ema \
#     --wandb \
#     --email 'KM BE-DISENT 0: CMNIST 1pct training done.\n CMNIST 2pct training...'

# python train.py \
#     --gpu_num 0 \
#     --train_disent_be \
#     --data_dir '/home/zungwooker/Debiasing/benchmarks' \
#     --device 'cuda' \
#     --dataset=cmnist \
#     --percent=2pct \
#     --lr=0.01 \
#     --exp=disent_be_gene_cmnist_2 \
#     --num_steps 50000 \
#     --num_bias_models 5 \
#     --agreement 3 \
#     --curr_step 10000 \
#     --lambda_swap 1 \
#     --lambda_dis_align 10 \
#     --lambda_swap_align 10 \
#     --use_lr_decay \
#     --lr_decay_step 10000 \
#     --lr_gamma 0.5 \
#     --projcode 'BE-DISENT-CMNIST' \
#     --run_name '2pct Origin+Gene, B.Net & BCDs get only original dataset' \
#     --preproc_dir '/home/zungwooker/Debiasing/preproc' \
#     --ema \
#     --wandb \
#     --email 'KM BE-DISENT 0: CMNIST 2pct training done.\n CMNIST 5pct training...'

# python train.py \
#     --gpu_num 0 \
#     --train_disent_be \
#     --data_dir '/home/zungwooker/Debiasing/benchmarks' \
#     --device 'cuda' \
#     --dataset=cmnist \
#     --percent=5pct \
#     --lr=0.01 \
#     --exp=disent_be_gene_cmnist_5 \
#     --num_steps 50000 \
#     --num_bias_models 5 \
#     --agreement 3 \
#     --curr_step 10000 \
#     --lambda_swap 1 \
#     --lambda_dis_align 10 \
#     --lambda_swap_align 10 \
#     --use_lr_decay \
#     --lr_decay_step 10000 \
#     --lr_gamma 0.5 \
#     --projcode 'BE-DISENT-CMNIST' \
#     --run_name '5pct Origin+Gene, B.Net & BCDs get only original dataset' \
#     --preproc_dir '/home/zungwooker/Debiasing/preproc' \
#     --ema \
#     --wandb \
#     --email 'KM BE-DISENT 0: CMNIST 5pct training done.\n CMNIST BE+DISENT+Ours Done!'
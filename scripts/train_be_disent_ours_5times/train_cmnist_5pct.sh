### LfF + BE

for i in {1..5}
do
python train.py \
    --gpu_num 0 \
    --train_disent_be_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=disent_be_gene_cmnist_5 \
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
    --projcode 'BE-DISENT-OURS-CMNIST' \
    --run_name '5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb
done
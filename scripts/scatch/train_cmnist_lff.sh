### LfF

python train.py \
    --gpu_num 3 \
    --train_lff_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=0.5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_0.5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'LFF-OURS-CMNIST' \
    --run_name '0.5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 3 \
    --train_lff_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=1pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_1 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'LFF-OURS-CMNIST' \
    --run_name '1pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 3 \
    --train_lff_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=2pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_2 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'LFF-OURS-CMNIST' \
    --run_name '2pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 3 \
    --train_lff_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'LFF-OURS-CMNIST' \
    --run_name '5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 3 \
    --train_lff_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=0.5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_0.5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'LFF-OURS-CMNIST' \
    --run_name '0.5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 3 \
    --train_lff_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=1pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_1 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'LFF-OURS-CMNIST' \
    --run_name '1pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 3 \
    --train_lff_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=2pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_2 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'LFF-OURS-CMNIST' \
    --run_name '2pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 3 \
    --train_lff_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'LFF-OURS-CMNIST' \
    --run_name '5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 3 \
    --train_lff_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=0.5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_0.5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'LFF-OURS-CMNIST' \
    --run_name '0.5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 3 \
    --train_lff_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=1pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_1 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'LFF-OURS-CMNIST' \
    --run_name '1pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 3 \
    --train_lff_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=2pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_2 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'LFF-OURS-CMNIST' \
    --run_name '2pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb

python train.py \
    --gpu_num 3 \
    --train_lff_ours \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'LFF-OURS-CMNIST' \
    --run_name '5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb
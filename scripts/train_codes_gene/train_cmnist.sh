### LfF + BE

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=0.5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_0.5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '0.5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 0.5pct training done.\n CMNIST 1pct training... | 1 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=1pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_1 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '1pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 1pct training done.\n CMNIST 2pct training... | 1 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=2pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_2 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '2pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 2pct training done.\n CMNIST 5pct training... | 1 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 5pct training done.\n CMNIST BE+Ours Done! | 1 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=0.5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_0.5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '0.5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 0.5pct training done.\n CMNIST 1pct training... | 2 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=1pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_1 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '1pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 1pct training done.\n CMNIST 2pct training... | 2 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=2pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_2 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '2pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 2pct training done.\n CMNIST 5pct training... | 2 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 5pct training done.\n CMNIST BE+Ours Done! | 2 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=0.5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_0.5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '0.5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 0.5pct training done.\n CMNIST 1pct training... | 3 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=1pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_1 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '1pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 1pct training done.\n CMNIST 2pct training... | 3 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=2pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_2 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '2pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 2pct training done.\n CMNIST 5pct training... | 3 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 5pct training done.\n CMNIST BE+Ours Done! | 3 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=0.5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_0.5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '0.5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 0.5pct training done.\n CMNIST 1pct training... | 4 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=1pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_1 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '1pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 1pct training done.\n CMNIST 2pct training... | 4 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=2pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_2 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '2pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 2pct training done.\n CMNIST 5pct training... | 4 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 5pct training done.\n CMNIST BE+Ours Done! | 4 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=0.5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_0.5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '0.5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 0.5pct training done.\n CMNIST 1pct training... | 5 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=1pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_1 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '1pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 1pct training done.\n CMNIST 2pct training... | 5 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=2pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_2 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '2pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 2pct training done.\n CMNIST 5pct training... | 5 iter'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cmnist \
    --percent=5pct \
    --lr=0.01 \
    --exp=lff_be_gene_cmnist_5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode 'BE-CMNIST' \
    --run_name '5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: CMNIST 5pct training done.\n CMNIST BE+Ours Done! | 5 iter'
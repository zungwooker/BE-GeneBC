### LfF + BE

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cifar10c \
    --percent=0.5pct \
    --lr=0.0005 \
    --exp=lff_be_gene_cifar10c_0.5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode BE-CIFAR10C-before-classFiltering \
    --run_name '0.5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: cifar10c 0.5pct training done.\n cifar10c 0.5pct lr0.0005 training...'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cifar10c \
    --percent=0.5pct \
    --lr=0.001 \
    --exp=lff_be_gene_cifar10c_0.5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode BE-CIFAR10C-before-classFiltering \
    --run_name '0.5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: cifar10c 0.5pct lr0.0005 training done.\n cifar10c 1pct training...'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cifar10c \
    --percent=1pct \
    --lr=0.001 \
    --exp=lff_be_gene_cifar10c_1 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode BE-CIFAR10C-before-classFiltering \
    --run_name '1pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: cifar10c 1pct training done.\n cifar10c 2pct training...'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cifar10c \
    --percent=2pct \
    --lr=0.001 \
    --exp=lff_be_gene_cifar10c_2 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode BE-CIFAR10C-before-classFiltering \
    --run_name '2pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: cifar10c 2pct training done.\n cifar10c 5pct training...'

python train.py \
    --gpu_num 0 \
    --train_lff_be \
    --data_dir '/home/zungwooker/Debiasing/benchmarks' \
    --device 'cuda' \
    --dataset=cifar10c \
    --percent=5pct \
    --lr=0.001 \
    --exp=lff_be_gene_cifar10c_5 \
    --num_steps 50000 \
    --num_bias_models 5 \
    --agreement 3 \
    --projcode BE-CIFAR10C-before-classFiltering \
    --run_name '5pct Origin+Gene, B.Net & BCDs get only original dataset' \
    --preproc_dir '/home/zungwooker/Debiasing/preproc' \
    --ema \
    --wandb \
    --email 'KM 0: cifar10c 5pct training done.\n cifar10c BE+Ours Done!'
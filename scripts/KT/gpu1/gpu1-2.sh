# BFFHQ vanilla 5pct

python train.py \
    --base \
    --train_vanilla \
    --dataset=bffhq \
    --percent=5pct \
    --lr=0.0001 \
    --data_dir '/home/work/human_pose/Debiasing/benchmarks' \
    --preproc_dir '/home/work/human_pose/Debiasing/preproc' \
    --projcode 'BFFHQ 5pct' \
    --run_name 'vanilla-base' \
    --device 'cuda' \
    --gpu_num 1 \
    --fix_randomseed \
    --seed 0 \
    --wandb
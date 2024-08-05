### LfF + BE
python train.py --train_lff_be --data_dir '/home/zungwooker/Debiasing/benchmarks' --device 'cuda' --dataset=bffhq --percent=0.5pct --lr=0.0001 --exp=lff_be_cmnist_0.5pct --num_steps 100000 --num_bias_models 5 --agreement 3 --projcode 'BE-testing-bffhq' --run_name 'test0(b for only origin)' --gpu_num 3 --preproc_dir '/home/zungwooker/Debiasing/preproc_youngoldperson_igs1.0' --wandb

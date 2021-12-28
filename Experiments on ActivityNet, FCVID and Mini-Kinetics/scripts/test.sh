#### actnet mbv2 p96,p128,p160
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py \
  --glance_arch mbv2 \
  --dataset actnet --data_dir /mnt/data2/170/actnet \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2 \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 96 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample --evaluate \
  --resume ckpt/actnet_p96_mbv2.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py \
  --glance_arch mbv2 \
  --dataset actnet --data_dir /mnt/data2/170/actnet \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample --evaluate \
  --resume ckpt/actnet_p128_mbv2.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py \
  --glance_arch mbv2 \
  --dataset actnet --data_dir /mnt/data2/170/actnet \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2 \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 160 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample --evaluate \
  --resume ckpt/actnet_p160_mbv2.pth.tar


#### actnet res50 p96,p128,p160
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py  \
  --glance_arch res50 --glance_ckpt_path ckpt/Initialization_model_prime_resnet50_patch_size_96_or_128.pth \
  --dataset actnet --data_dir /mnt/data2/170/actnet \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 96 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample --evaluate \
  --resume ckpt/actnet_p96_res50.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py  \
  --glance_arch res50 --glance_ckpt_path ckpt/Initialization_model_prime_resnet50_patch_size_96_or_128.pth \
  --dataset actnet --data_dir /mnt/data2/170/actnet \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample --evaluate \
  --resume ckpt/actnet_p128_res50.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py  \
  --glance_arch res50 --glance_ckpt_path ckpt/Initialization_model_prime_resnet50_patch_size_96_or_128.pth \
  --dataset actnet --data_dir /mnt/data2/170/actnet \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 160 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample --evaluate \
  --resume ckpt/actnet_p160_res50.pth.tar

### fcvid mbv2 p128
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py \
  --glance_arch mbv2 \
  --dataset fcvid --data_dir /mnt/data2/fcvid \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample --evaluate \
  --resume ckpt/fcvid_p128_mbv2.pth.tar

### fcvid res50 p128

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py  \
  --glance_arch res50 --glance_ckpt_path ckpt/Initialization_model_prime_resnet50_patch_size_96_or_128.pth \
  --dataset fcvid --data_dir /mnt/data2/fcvid \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample --evaluate \
  --resume ckpt/fcvid_p128_res50.pth.tar


#### minik mbv2 p128
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py \
  --glance_arch mbv2 \
  --dataset minik --data_dir /home/data2/kinetics400_dataset/kinetics400_1/kinetics400 \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.003 --lr_type cos --lr_steps 0 0 \
  --batch_size 48 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample --evaluate \
  --resume ckpt/minik_p128_mbv2.pth.tar

#### minik res50 p128
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py \
  --glance_arch res50 --glance_ckpt_path ckpt/Initialization_model_prime_resnet50_patch_size_96_or_128.pth \
  --dataset minik --data_dir /home/data2/kinetics400_dataset/kinetics400_1/kinetics400 \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.003 --lr_type cos --lr_steps 0 0 \
  --batch_size 48 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample --evaluate \
  --resume ckpt/minik_p128_res50.pth.tar
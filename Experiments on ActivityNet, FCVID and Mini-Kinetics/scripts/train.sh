#### actnet mbv2 p96,p128,p160
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py \
  --glance_arch mbv2 \
  --dataset actnet --data_dir PATH_TO_DATASET \
  --root_log LOG_DIR \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2 \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 96 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py \
  --glance_arch mbv2 \
  --dataset actnet --data_dir PATH_TO_DATASET \
  --root_log LOG_DIR \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py \
  --glance_arch mbv2 \
  --dataset actnet --data_dir PATH_TO_DATASET \
  --root_log LOG_DIR \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2 \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 160 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample


#### actnet res50 p96,p128,p160
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py  \
  --glance_arch res50 --glance_ckpt_path ckpt/Initialization_model_prime_resnet50_patch_size_96.pth \
  --dataset actnet --data_dir PATH_TO_DATASET \
  --root_log LOG_DIR \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 96 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py  \
  --glance_arch res50 --glance_ckpt_path ckpt/Initialization_model_prime_resnet50_patch_size_96.pth \
  --dataset actnet --data_dir PATH_TO_DATASET \
  --root_log LOG_DIR \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py  \
  --glance_arch res50 --glance_ckpt_path ckpt/Initialization_model_prime_resnet50_patch_size_96.pth \
  --dataset actnet --data_dir PATH_TO_DATASET \
  --root_log LOG_DIR \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 160 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample


### fcvid mbv2 p128
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py \
  --glance_arch mbv2 \
  --dataset fcvid --data_dir PATH_TO_DATASET \
  --root_log LOG_DIR \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample

### fcvid res50 p128

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py  \
  --glance_arch res50 --glance_ckpt_path ckpt/Initialization_model_prime_resnet50_patch_size_96.pth \
  --dataset fcvid --data_dir PATH_TO_DATASET \
  --root_log LOG_DIR \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample


#### minik mbv2 p128
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py \
  --glance_arch mbv2 \
  --dataset minik --data_dir PATH_TO_DATASET \
  --root_log LOG_DIR \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.003 --lr_type cos --lr_steps 0 0 \
  --batch_size 48 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample

#### minik res50 p128
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py \
  --glance_arch res50 --glance_ckpt_path ckpt/Initialization_model_prime_resnet50_patch_size_96.pth \
  --dataset minik --data_dir PATH_TO_DATASET \
  --root_log LOG_DIR \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.003 --lr_type cos --lr_steps 0 0 \
  --batch_size 48 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample
# Experiments on ActivityNet, FCVID and Mini-Kinetics

## Requirements

- python 3.8
- pytorch 1.8.0
- torchvision 0.8.0

## Datasets
1. Please get the train/test split files for each dataset from [Google Drive](https://drive.google.com/drive/folders/1QZ2gVoGMh3Xe20kgdG6s7cnQTYsDWQXz?usp=sharing) and put them in `PATH_TO_DATASET`.
2. Download videos from following links, or contact the corresponding authors for the access. 

   - [ActivityNet-v1.3](http://activity-net.org/download.html) 
   - [FCVID](https://drive.google.com/drive/folders/1cPSc3neTQwvtSPiVcjVZrj0RvXrKY5xj)
   - [Mini-Kinetics](https://deepmind.com/research/open-source/kinetics). Please download [Kinetics 400](https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz) and select the 200 classes for minik.

3. Extract frames using [ops/video_jpg.py](ops/video_jpg.py). Minor modifications on file path are needed when extracting frames from different dataset. You may also need to change dataset config in `dataset_config.py`  as well.

## Training

To train AdaFocusV2 with patch size 128 and mobilenet_v2 as global CNN:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py \
  --glance_arch mbv2 \
  --dataset actnet --data_dir PATH_TO_DATASET \
  --root_log LOG_DIR \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample
```

To use res50 as global CNN, a res50 model pretrained on ImageNet with image size 96 is required, which can be download from here. (file name: `Initialization_model_prime_resnet50_patch_size_96_or_128.pth`)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py  \
  --glance_arch res50 --glance_ckpt_path RES50_INIT_PATH \
  --dataset actnet --data_dir PATH_TO_DATASET \
  --root_log LOG_DIR \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample
```

To get all training commands, please refer to `scripts/train.sh`

## Pre-trained Models

We provide pre-trained models for multiple patch sizes and datasets. All pre-trained models are available at [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/5f686486750f4cf6a11a/) / [Google Drive]()

### ActivityNet

| file name                 | Arch                | Patch Size | mAP  | GFLOPS |
| :------------------------ | ------------------- | ---------- | ---- | ------ |
| actnet_p96_mbv2.pth.tar   | MobileNetV2 + Res50 | 96         | 77.4 | 17.6   |
| actnet_p128_mbv2.pth.tar  | MobileNetV2 + Res50 | 128        | 79.0 | 27.0   |
| actnet_p160_mbv2.pth.tar  | MobileNetV2 + Res50 | 160        | 79.9 | 39.0   |
| actnet_p96_res50.pth.tar  | Res50               | 96         | 77.5 | 24.8   |
| actnet_p128_res50.pth.tar | Res50               | 128        | 78.9 | 34.1   |
| actnet_p160_res50.pth.tar | Res50               | 160        | 80.2 | 46.1   |

### FCVID

| file name                | Arch                | Patch Size | mAP  | GFLOPS |
| ------------------------ | ------------------- | ---------- | ---- | ------ |
| fcvid_p128_mbv2.pth.tar  | MobileNetV2 + Res50 | 128        | 85.0 | 27.0   |
| fcvid_p128_res50.pth.tar | Res50               | 128        | 84.5 | 34.1   |

### Mini-Kinetics

| file name                | Arch                | Patch Size | Acc1 | GFLOPS |
| ------------------------ | ------------------- | ---------- | ---- | ------ |
| minik_p128_mbv2.pth.tar  | MobileNetV2 + Res50 | 128        | 75.4 | 27.0   |
| minik_p128_res50.pth.tar | Res50               | 128        | 74.0 | 34.1   |

To evaluate pretrained models, please download the models (as well as the initialization model for res50) and save to the folder `ckpt`. Commands for reproducing results in the paper are in `scripts/test.sh`.

For example,  here's the command to evaluate `actnet_p128_mbv2.pth.tar`

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_v2.py \
  --glance_arch mbv2 \
  --dataset actnet --data_dir PATH_TO_DATASET \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample --evaluate \
  --resume ckpt/actnet_p128_mbv2.pth.tar
```

## Early Exit

To Test AdaFocusV2+ (Early Exiting), run the script `test_early_exit.py` with the same arguments. E.g.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python test_early_exit.py \
  --glance_arch mbv2 \
  --dataset actnet --data_dir PATH_TO_DATASET \
  --workers 8 --num_segments 16 --dropout 0.2  --fc_dropout 0.2  \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0 \
  --batch_size 32 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.1 --classifier_lr_ratio 5.0 \
  --hidden_dim 1024 --stn_hidden_dim 2048 --sample --evaluate \
  --resume ckpt/actnet_p128_mbv2.pth.tar
```

## Acknowledgement

- We use the implementation of MobileNet-v2 and ResNet from [Pytorch source code](https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html). 
- Feature aggregation modules are borrowed from [FrameExit](https://github.com/Qualcomm-AI-research/FrameExit).
- We borrow some codes for dataset preparation from [AR-Net](https://github.com/mengyuest/AR-Net#dataset-preparation)

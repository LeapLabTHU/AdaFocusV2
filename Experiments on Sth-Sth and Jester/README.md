# Experiments on Something-Something V1&V2 and Jester

## Requirements
- python 3.8
- pytorch 1.8.0
- torchvision 0.8.0


## Datasets
- Please follow the instruction of [TSM](https://github.com/mit-han-lab/temporal-shift-module#data-preparation) to prepare the Something-Something V1/V2 and Jester datasets.

- After preparation, please edit the "ROOT_DATASET" in [ops/dataset_config.py](ops/dataset_config.py) to the correct path of the dataset.


## Pre-trained Models

- TODO


## Training

Run the following command. Note that the dataset and patch size should be specified (i.e., at [something/somethingv2/jester] and [128/144/160/176])
```

python main.py [something/somethingv2/jester] RGB \
     --root_log PATH_TO_SAVE_LOG  --root_model PATH_TO_SAVE_CKPT \
     --arch resnet50 --num_segments 8 \
     --gd 20 -j 8 --lr_steps 0 0 --lr_type cos --epochs 50 \
     --batch-size 32 --lr 0.01 --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb --print-freq=1 \
     --patch_size [128/144/160/176] --global_lr_ratio 0.5 --stn_lr_ratio 0.01

```


## Evaluate Pre-trained Models
- Add "--evaluate" to the training script for the evaluation mode.
- Add "--resume PATH_TO_CKPT" to the training script to specify the pre-trained models.

## Acknowledgement
Our code is based on the official implementation of [temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-module).

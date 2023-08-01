# Train Your Model

## Script For Training

Our training pipeline is based on [mmcv](https://github.com/open-mmlab/mmcv) and [mmdet3d](https://github.com/open-mmlab/mmdetection3d). To train a DriveAdapter model, you could use:
```shell
#In DriveAdapter/open_loop_training/ directory
#We train on 16 A100 for 4 days
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7, python -m torch.distributed.launch --nproc_per_node=8 --master_port=22023 train.py configs/driveadapter.py --work-dir=work_dirs/driveadapter --launcher="pytorch"
```

For single GPU debug, you could simply use:
```shell
#In driveadapter/open_loop_training/ directory
CUDA_VISIBLE_DEVICES=0 python train.py configs/driveadapter.py --work-dir=work_dirs/debug
```

## Code Structure
We give the structure of the training code. Note that We only introduce those folders/files are commonly used and modified.

    DriveAdapter/open_loop_training
    ├── ckpt                    # Checkpoints
    ├── configs                 # Hyper-Parameter
    ├── work_dirs               # Training Log
    ├── code                    # Preprocessing, DataLoader, Model
    │   ├── apis                    # Training pipeline for mmdet3D
    │   ├── core                    # The hooks for mmdet3D
    │   ├── datasets                # Preprocessing and DataLoader
    |   |   ├── pipelines                # Functions of Preprocessing and DataLoader
    │   |   ├── samplers                 # For DDP
    │   |   └── carla_dataset.py         # Framework of Preprocessing and DataLoading
    │   ├── model_code                   # Neural Network
    |   |   ├── backbones                # Module of Encoder
    |   |   └── dense_heads              # Module of Decoder and Loss Functions
    │   └── encoder_decoder_framework.py # Entrance of Neural Network
    └── train.py                # Entrance of Training



## Tips
- Change **is_dev** in [open_loop_training/configs/driveadapter.py](../open_loop_training/configs/driveadapter.py) to True when you develop your model and to False during training
- Set **is_full** in [open_loop_training/configs/driveadapter.py](../open_loop_training/configs/driveadapter.py) to False would only use the same number of data as TCP while to True would use all possible data recorded in **dataset/dataset_metadata.pkl**.
- Your could start with [open_loop_training/code/encoder_decoder_framework.py](../open_loop_training/configs/driveadapter.py) when you want to learn about the neural network and [open_loop_training/code/datasets/carla_dataset.py](../open_loop_training/configs/driveadapter.py) when you want to learn about data.
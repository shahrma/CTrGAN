# CTrGAN: Cycle Transformers GAN for Gait Transfer



This repository contains training code for the examples in the WACV 2023 paper "[CTrGAN: Cycle Transformers GAN for Gait Transfer
](https://www.gil-ba.com/ctrgan/CTrGAN.html)."


CTrGAN transfers the poses of unseen source to the target, while maintaining the natural gait of the target. 
From left to right: (a) The source’s image is converted to 
(b) [DensePose](https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/README.md)’s  IUV format.
(c) The **source** pose is rendered (using [Vid2Vid](https://github.com/NVIDIA/vid2vid)) to a corresponding RGB image of the target.
(c) Our model translates the IUV of the source to the corresponding most natural IUV pose of the target by synthesizing a novel pose. 
(e) The **generated** pose is rendered (using [Vid2Vid](https://github.com/NVIDIA/vid2vid)) to a corresponding RGB image of the target.

<img src='./images/0007_T0004-W-WO.gif' height="160px"/>
<img src='./images/titles.jpg' height="64px"/>

**Note :** The code in this repository includes only the CTrGAN code that we used to convert the IUV Pose. 
It does not include the part that we used to create the Source Pose from the object appearance ([Detectron2-Densepose](https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/README.md)), 
and the part that we used to convert back from Pose to Appearance ([Vid2Vid](https://github.com/NVIDIA/vid2vid)))

## Prerequisites
- Linux
- Python 3.8+
- NVIDIA GPU + CUDA CuDNN

The files requirements.txt environment.yml show the packages we use in the runtime environment.
## Data Processing
The code expects to receive iuv images according to the structure shown in ./example/dataset/iuv . 
Each image will be a png cropped to 256x256 pixels. 
The object should sit in the center of the image. 
An image consists of 4 iuv layers and a mask.

## Training
### Example
```code 
python train.py  --name EXP5010 --datafile ./example/configs/train_example.yaml --checkpoints_dir ./example/ --model CTrGAN --ngf 16 --dataset_mode unaligned_sequence --no_dropout --no_flip --loadSize 272 --fineSize 256 --iuv_mode iuv1 --input_nc 4 --output_nc 4 --use_perceptual_loss --pool_size 0 --niter 100 --niter_decay 300 --save_epoch_freq 100 --continue_train --epoch_count 0 --use_sa --use_qsa --seq_len 3 --nThreads 0 --gpu 0
``` 

## Inference
### Example
Note: Be sure to update the variable which_epoch to the value with which you wish to perform the predict. (If the code does not find the model, it will use the initialized parameters of the model, i.e. random)

```code 
python predict.py --name EXP5010 --datafile ./example/configs/valid_example.yaml --checkpoints_dir ./example/ --model CTrGAN --ngf 16 --dataset_mode unaligned_sequence --no_dropout --no_flip --loadSize 256 --fineSize 256 --iuv_mode iuv1 --input_nc 4 --output_nc 4 --which_epoch 80 --results_dir ./example/results/ --use_fullseq --seq_len 3 --use_sa --use_qsa --gpu 0
```
## Acknowledgments
This code is based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [Recycle-GAN :Unsupervised Video Retargeting](https://github.com/aayushbansal/Recycle-GAN).
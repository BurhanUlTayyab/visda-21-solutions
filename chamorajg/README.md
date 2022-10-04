## Train Model

#### Train a Base / Teacher Model
1.  `./distributed_train.sh 2 imagenet_folder_dir --model tf_efficientnet_b7 --num-classes 1000 --sched cosine --epochs 250 --opt 'adam' --warmup-epochs 5 --lr 0.01 --model-ema --drop-connect 0.2 -b 64 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5`

#### Get Pseudo Labels and Sample them.
1. `python test.py --test_dir TESTDIR --result_dir RESULTDIRNAME --result_fname RESULTFNAME --checkpoint CHECKPOINTPATH`
2. `python pseudosampler.py --src_dir TESTDIR --dest_dir IMAGENETDIR --samples 1000` 
samples can be 1500 as well.

#### Train a Student on Noisy Data using Pseudo Labels.
1. `./distributed_train.sh 2 imagenet_folder_dir --model tf_efficientnet_b7 --num_classes 1000 --sched cosine --epochs 250 --opt 'adam' --warmup-epochs 5 --lr 0.01 --reprob 0.5 --hflip 0.5 --vflip 0.5 --mixup 0.25 --cutmix 0.5 --mixup-prob 0.5 --bce-loss --drop 0.4 --remode pixel --model-ema --drop-connect 0.2 -b 64 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5`

#### Info
Can repeat the student training of the model 2-3 times to improve the accuracy.
Increase the sampling size to increase the size of the pseudo labelled images in the train dataset.
## References
Large part of the code was adapted from [timm](https://github.com/rwightman/pytorch-image-models)
## Pretraining
### To start pretraining
```
python3 pretrain.py
```
You can change the DATASET_PATH in the file to the appropriate dataset (Line 27)

After pretraining you can finetune the model by

```
# Set the path to save checkpoints
OUTPUT_DIR=/path/to/save/your_model
DATA_PATH=/path/to/imagenet_path

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
    --model beit_base_patch16_224 --data_path /path/to/imagenet \
    --finetune $PRETRAINED_PATH \
    --output_dir /path/to/save_result --batch_size 64 --lr 2e-5 --update_freq 1 \
    --warmup_epochs 5 --epochs 30 --layer_decay 0.85 --drop_path 0.1 \
    --weight_decay 1e-8 --enable_deepspeed
```

To test

```
python test.py
```

The trained checkpoints could be found here: https://drive.google.com/file/d/1WXhMWtTMKz5-XXwwjZN8zdP9xUiP5v8Z/view?usp=sharing
## References

[BEiT: BERT Pre-Training of Image Transformers](https://github.com/microsoft/unilm/tree/master/beit)


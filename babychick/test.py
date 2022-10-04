# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import torch
import torch.backends.cudnn as cudnn
from functools import partial
from modeling_finetune import VisionTransformer
from dataset_folder import ImageFolder
from torchvision import transforms


def main():
    IMG_SIZE = 384
    DATASET_PATH = "./1/val"
    MEAN, STD = 0.5, 0.5
    BATCH_SIZE = 64
    BEIT_MODEL = "beit_base_patch16_384"
    CHECKPOINT_PATH = "visda.pth"


    device = torch.device("cuda")
    cudnn.benchmark = True

    dataset_val = ImageFolder(DATASET_PATH, transform=transforms.Compose([transforms.Resize(IMG_SIZE, interpolation=3),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
        ]))


    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=BATCH_SIZE
        )

    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_classes=1000, in_chans=3, drop_rate=0.0, 
        drop_path_rate=0.1, attn_drop_rate=0.0, use_mean_pooling=True,
        init_scale=0.001, use_rel_pos_bias=True, use_abs_pos_emb=False, init_values=0.2)


    model.to(device)
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint)

    for batch in data_loader_val:
        image = batch[0]
        image = image.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            output = model(image)
            print(output.topk(1))



if __name__ == '__main__':
    main()

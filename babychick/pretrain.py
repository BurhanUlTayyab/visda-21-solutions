import torch
import torch.nn as nn

from modeling_pretrain import VisionTransformerForMaskedImageModeling
from functools import partial
from datasets import DataAugmentationForBEiT
from dataset_folder import ImageFolder
from torchvision import datasets, transforms
from optim_factory import get_parameter_groups
from utils import NativeScalerWithGradNormCount as NativeScaler
from engine_for_pretraining import train_one_epoch
from timm.utils import get_state_dict
import utils

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 1000
WINDOW_SIZE = (16, 16)
NUM_PATCHES = 75
IMG_SIZE = 224
SECOND_INPUT_SIZE = 112
BATCH_SIZE = 1
NUM_WORKERS = 4
MAX_MASK_PATCHES_PER_BLOCK = None
MIN_MASK_PATCHES_PER_BLOCK = 16
DISCRETE_VAE_TYPE = "dall-e"
DATASET_PATH = "./1/val"
DISCRETE_VAE_WEIGHT_PATH = "./tokenizer"
DEVICE = "cuda"
WEIGHT_DECAY = 0.05
WEIGHT_DECAY_END = None
OPT = "adamw"
LEARNING_RATE = 0.0015
MIN_LEARNING_RATE = 1e-5
EPOCHS = 800
WARMUP_EPOCHS = 10
CLIP_GRAD = 3.0
SAVE_CKPT_FREQ = 2000
SAVE_DIR = "./output_dir"


def save_model(save_dir, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = save_dir
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)
            torch.save(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)

model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, num_classes = NUM_CLASSES, 
        in_chans= 3, drop_path_rate = 0.1, use_shared_rel_pos_bias= True, use_abs_pos_emb = False, init_values =  0.1)

transform = DataAugmentationForBEiT(MEAN, STD, IMG_SIZE, WINDOW_SIZE, NUM_PATCHES, MAX_MASK_PATCHES_PER_BLOCK, MIN_MASK_PATCHES_PER_BLOCK, DISCRETE_VAE_TYPE)
dataset_train = ImageFolder(DATASET_PATH, transform=transform)

d_vae = utils.create_d_vae(
    weight_path=DISCRETE_VAE_WEIGHT_PATH, d_vae_type=DISCRETE_VAE_TYPE,
    device=DEVICE, image_size=SECOND_INPUT_SIZE)

sampler_train = torch.utils.data.RandomSampler(dataset_train)

data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        sampler=sampler_train
    )

model.to(DEVICE)
model_without_ddp = model
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_training_steps_per_epoch = len(dataset_train) // BATCH_SIZE

if WEIGHT_DECAY:    
    skip = model.no_weight_decay()
    parameters = get_parameter_groups(model, WEIGHT_DECAY, skip, None, None)
    weight_decay = 0.
    opt_args = {'lr': LEARNING_RATE, 'weight_decay': weight_decay, 'eps': 1e-08, 'betas': [0.9, 0.999]}
    optimizer = torch.optim.AdamW(parameters, **opt_args)
    loss_scaler = NativeScaler()

lr_schedule_values = utils.cosine_scheduler(
    LEARNING_RATE, MIN_LEARNING_RATE, EPOCHS, num_training_steps_per_epoch,
    warmup_epochs=WARMUP_EPOCHS, warmup_steps=-1,
)

if WEIGHT_DECAY_END is None:
    WEIGHT_DECAY_END = WEIGHT_DECAY
wd_schedule_values = utils.cosine_scheduler(
    WEIGHT_DECAY, WEIGHT_DECAY_END, EPOCHS, num_training_steps_per_epoch)

for epoch in range(0, EPOCHS):
    train_stats = train_one_epoch(
            model, d_vae, data_loader_train,
            optimizer, DEVICE, epoch, loss_scaler,
            CLIP_GRAD, log_writer=None,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
    )

    if (epoch + 1) % SAVE_CKPT_FREQ == 0 or epoch + 1 == EPOCHS:
        save_model(
            save_dir=SAVE_DIR, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=epoch)




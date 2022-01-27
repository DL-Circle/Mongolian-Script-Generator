#-*- coding:utf-8 -*-

from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from dataset import ScriptImageGenerator
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--datafolder', type=str)
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=2)
parser.add_argument('--crop_size', type=int, default=200)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--epochs', type=int, default=50000)
parser.add_argument('--timesteps', type=int, default=1000)
args = parser.parse_args()

datafolder = args.datafolder
input_size = args.input_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks

transform = Compose([
    ToPILImage(),
    RandomCrop(args.crop_size),
    Resize(input_size),
    ToTensor(),
    Lambda(lambda t: (t * 2) - 1)
])
dataset = ScriptImageGenerator(datafolder, input_size, transform=transform)
model = create_model(input_size, num_channels, num_res_blocks).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    timesteps = args.timesteps,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    dataset,
    image_size = input_size,
    train_batch_size = args.batchsize,
    train_lr = 2e-5,
    train_num_steps = args.epochs,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = True                       # turn on mixed precision training with apex
)

trainer.train()

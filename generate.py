#-*- coding:utf-8 -*-

from diffusion_model.trainer import GaussianDiffusion, num_to_groups
from diffusion_model.unet import create_model
from torchvision import utils
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exportfile', type=str, default='sample.png')
parser.add_argument('-w', '--weightfile', type=str)
parser.add_argument('-d', '--device', type=str, default='cuda')
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=2)
parser.add_argument('--timesteps', type=int, default=1000)
parser.add_argument('-s', '--num_sample', type=int, default=16)
args = parser.parse_args()

exportfile = args.exportfile
weightfile = args.weightfile
input_size = args.input_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_sample = args.num_sample
device = args.device

model = create_model(input_size, num_channels, num_res_blocks)

diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    timesteps = args.timesteps,   # number of steps
    loss_type = 'l1'    # L1 or L2
).to(device)

weight = torch.load(weightfile)
diffusion.load_state_dict(weight['ema'])
print("Model Loaded!")

batches = num_to_groups(4, num_sample)
imgs_list = list(map(lambda n: diffusion.sample(batch_size=num_sample), batches))
imgs = torch.cat(imgs_list, dim=0)
imgs = (imgs + 1) * 0.5
utils.save_image(imgs, exportfile, nrow = 4)
print("Done!")
# import matplotlib.pyplot as plt
# plt.imshow(imgs)
# plt.show()

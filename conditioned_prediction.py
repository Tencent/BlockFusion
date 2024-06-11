import tqdm
from diffusers import DiffusionPipeline
import torch
import numpy as np
import os
from diffusers.utils.torch_utils import randn_tensor
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--layout', type=str, default='samplelayouts/exp1_32-56-24/0_0.npy',help='path to single block layout')
parser.add_argument('--save', type=str, default='output/cond',help='save dir')
args = parser.parse_args()

pipeline = DiffusionPipeline.from_pretrained("checkpoints/conditioned", torch_dtype=torch.float32)
pipeline.to("cuda")
generator = torch.Generator(device=pipeline.device).manual_seed(22222)
for v in pipeline.unet.parameters():
    v.requires_grad=False
batch_size = 1
num_inference_steps =1000
image_shape = (batch_size, pipeline.unet.config.in_channels, *pipeline.unet.config.sample_size)
images = randn_tensor(image_shape, generator=generator, device=pipeline.device)

layoutlatent = np.zeros((batch_size,9,32,96),dtype=np.float32)
layoutlatent[:,:,:,32:64] = np.load(args.layout).astype(np.float32)
layoutlatent = torch.from_numpy(layoutlatent).to(pipeline.device)

# load layout

pipeline.scheduler.set_timesteps(num_inference_steps)
images[:, 2:, :, :] = layoutlatent
with torch.autocast('cuda'):
    for t in tqdm.tqdm(pipeline.scheduler.timesteps):
        model_output = pipeline.unet(images, t).sample
        images = pipeline.scheduler.step(model_output, t, images[:,:2,:,:], generator=generator).prev_sample
        if t>0:
            images = torch.concatenate((images,layoutlatent),dim=1)

images = images.cpu().numpy()*1.42
latent_dir = os.path.join(args.save, "latents")
os.makedirs(latent_dir ,exist_ok=True)
for i,image in enumerate(images):
    np.save(os.path.join(latent_dir,str(i)+'.npy'),image[np.newaxis,:])

from decode import test_vae_sdf

output_dir = os.path.join(args.save, "mesh")
os.makedirs(output_dir ,exist_ok=True)
test_vae_sdf(latent_dir,output_dir)
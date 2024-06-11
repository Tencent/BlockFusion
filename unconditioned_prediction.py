from diffusers import DiffusionPipeline
import torch
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=4, help='x scale in in decimeter')
parser.add_argument('--save', type=str, default='output/uncond',help='save dir')
args = parser.parse_args()

pipeline = DiffusionPipeline.from_pretrained("checkpoints/unconditioned", torch_dtype=torch.float32)
pipeline.to("cuda")
generator = torch.Generator(device=pipeline.device).manual_seed(75)
images = pipeline(
    generator=generator,
    batch_size=args.batch,
    num_inference_steps=1000,
    output_type="numpy",
).images

images = images.transpose(0,3,1,2)
images = images*1.42
latent_dir = os.path.join(args.save, "latents")
os.makedirs(latent_dir ,exist_ok=True)
for i,image in enumerate(images):
    np.save(os.path.join(latent_dir ,str(i)+'.npy'),image[np.newaxis,:])

# load uncondtional pipeline and run inference

from decode import test_vae_sdf

output_dir = os.path.join(args.save, "mesh")
os.makedirs(output_dir ,exist_ok=True)
test_vae_sdf(latent_dir,output_dir)
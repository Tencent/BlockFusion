import tqdm
from diffusers import DiffusionPipeline,RePaintScheduler,RePaintPipeline
import torch
import numpy as np
import os
from diffusers.utils.torch_utils import randn_tensor
from math import ceil
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--layout', type=str, default='samplelayouts/exp1_32-56-24', help='path to layout directory')
parser.add_argument('--resample', type=int, default=15, help='times of resample')
parser.add_argument('--save', type=str, default='output',help='save dir')
args = parser.parse_args()
# slide window
def get_views(panorama_height, panorama_width, window_size=64, stride=32):
    num_blocks_height = ceil((panorama_height - window_size) / stride) + 1
    num_blocks_width = ceil((panorama_width - window_size) / stride) + 1
    views = [[] for _ in range(num_blocks_height)]
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            h_start = int(i  * stride)
            h_end = h_start + window_size
            w_start = int(j  * stride)
            w_end = w_start + window_size
            views[i].append((h_start, h_end, w_start, w_end))
    return views

layoutdir =  args.layout
expname = layoutdir.split('/')[-1]
outputdir = os.path.join(args.save,expname)
os.makedirs(outputdir, exist_ok=True)
shutil.copytree(layoutdir, os.path.join(outputdir,'layout_'+ layoutdir.split('/')[-1].split('_')[1] ))
x,z,stride = layoutdir.split('_')[1].split('-')
x,z,stride = int(x),int(z),int(stride)
y = 32 #fixed
target_vol_shape = (x,y,z)
views = get_views(x, z,window_size=32,stride=stride)
num_views = len(views)*len(views[0])

layoutlatent = np.zeros((num_views,9,32,96),dtype=np.float32)
for i in range(len(views)):
    for j in range(len(views[0])):
        layoutlatent[i * len(views[0]) + j,:,:,32:64]  = np.load(os.path.join(layoutdir,'{}_{}.npy'.format(i,j))).astype(np.float32)

# devide slide window
# prepare layout
in_channel = 2
num_inference_steps = 1000

pipeline = DiffusionPipeline.from_pretrained("checkpoints/conditioned", torch_dtype=torch.float32)
pipeline.to("cuda")
device = 'cuda'
generator = torch.Generator(device=pipeline.device).manual_seed(99)
for v in pipeline.unet.parameters():
    v.requires_grad=False
image_shape = (num_views, in_channel, *pipeline.unet.config.sample_size)
images = randn_tensor(image_shape, generator=generator, device=pipeline.device)
layoutlatent = torch.from_numpy(layoutlatent).to(pipeline.device)
# prepare noise

pipeline.scheduler.set_timesteps(num_inference_steps)
with torch.autocast('cuda'):
    for t in tqdm.tqdm(pipeline.scheduler.timesteps):
        images = torch.cat((images, layoutlatent), dim=1)
        model_output = pipeline.unet(images, t).sample
        images = pipeline.scheduler.step(model_output, t, images[:,:in_channel,:,:], generator=generator).prev_sample
images = images*1.42
# infer without considering adjacency relevance

x = 32 + (len(views)-1)*stride
z = 32 + (len(views[0])-1)*stride
latent1 = torch.randn((len(views[0]), in_channel, x, y), device=device)
latent2 = torch.randn((1, in_channel, x, z), device=device)
latent3 = torch.randn((len(views), in_channel, z, y), device=device)
noise = torch.randn((num_views, in_channel, 32, 96), device=device)

original_image = torch.zeros_like(noise)
mask_image = torch.zeros_like(noise)

for i in range(len(views)):
    for j in range(len(views[0])):
        if i % 2 == 0 and j % 2 == 0:
            original_image[i * len(views[0]) + j, :, :, :] = images[i * len(views[0]) + j]
            mask_image[i*len(views[0])+j, :, :, :] = 1

# load latent at intervals

del pipeline,images
torch.cuda.empty_cache()

generator = torch.Generator(device='cuda').manual_seed(7)
scheduler = RePaintScheduler.from_pretrained("checkpoints/conditioned/scheduler")
pipeline = RePaintPipeline.from_pretrained("checkpoints/conditioned", scheduler=scheduler)
for v in pipeline.unet.parameters():
    v.requires_grad=False
device = "cuda"
pipeline = pipeline.to(device)
weight = 1000000 #very large number, make it easy to do repaint

count1 = torch.zeros_like(latent1)
value1 = torch.zeros_like(latent1)
count2 = torch.zeros_like(latent2)
value2 = torch.zeros_like(latent2)
count3 = torch.zeros_like(latent3)
value3 = torch.zeros_like(latent3)


num_inference_steps=1000
jump_length=100 # steps of resample
jump_n_sample=args.resample # times of resample
generator=generator
scheduler.set_timesteps(num_inference_steps, jump_length, jump_n_sample, device)
t_last = scheduler.timesteps[0] + 1

with torch.autocast('cuda'):
    for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
        if t < t_last:
            noise = torch.cat((noise,layoutlatent),dim=1)
            count1.zero_()
            value1.zero_()
            count2.zero_()
            value2.zero_()
            count3.zero_()
            value3.zero_()
            latent1.zero_()
            latent2.zero_()
            latent3.zero_()
            model_output = pipeline.unet(noise, t).sample
            noise = scheduler.step(model_output, t, noise[:,:in_channel,:,:], original_image, mask_image, generator).prev_sample
            for i in range(len(views)):
                for j in range(len(views[0])):
                    if i%2==0 and j%2==0:
                        x_start, x_end, z_start, z_end = views[i][j]
                        noise1, noise2, noise3 = noise[i * len(views[0]) + j].split(32, dim=-1)
                        value1[j, :, x_start:x_end, :] += weight*noise1
                        count1[j, :, x_start:x_end, :] += weight
                        value2[:, :, x_start:x_end, z_start:z_end] += weight*noise2
                        count2[:, :, x_start:x_end, z_start:z_end] += weight
                        value3[i, :, z_start:z_end, :] += weight*noise3
                        count3[i, :, z_start:z_end, :] += weight
                    else:
                        x_start, x_end, z_start, z_end = views[i][j]
                        noise1, noise2, noise3 = noise[i * len(views[0]) + j].split(32, dim=-1)
                        value1[j, :, x_start:x_end, :] += noise1
                        count1[j, :, x_start:x_end, :] += 1
                        value2[:, :, x_start:x_end, z_start:z_end] += noise2
                        count2[:, :, x_start:x_end, z_start:z_end] += 1
                        value3[i, :, z_start:z_end, :] += noise3
                        count3[i, :, z_start:z_end, :] += 1
            latent1 = torch.where(count1 > 0, value1 / count1, value1)
            latent2 = torch.where(count2 > 0, value2 / count2, value2)
            latent3 = torch.where(count3 > 0, value3 / count3, value3)
            for i in range(len(views)):
                for j in range(len(views[0])):
                    x_start, x_end, z_start, z_end = views[i][j]
                    noise[i*len(views[0])+j, :, :, :32] = latent1[j,:,x_start:x_end, :]
                    noise[i*len(views[0])+j, :, :,32:64] = latent2[:,:,x_start:x_end, z_start:z_end]
                    noise[i*len(views[0])+j, :, :,64:] = latent3[i,:,z_start:z_end,:]
        else:
            noise = scheduler.undo_step(noise, t_last, generator)
        t_last = t
images = noise.cpu().numpy()
images = images*1.42
latent_dir = os.path.join(outputdir,'latents')
os.makedirs(latent_dir,exist_ok=True)
for i in range(len(views)):
    for j in range(len(views[0])):
        image = images[i*len(views[0])+j]
        np.save(os.path.join(latent_dir,str(i)+'_'+str(j)+'.npy'),image[np.newaxis,:])
# save

from decode import test_vae_sdf

meshdir = os.path.join(outputdir,'mesh')
os.makedirs(meshdir ,exist_ok=True)
test_vae_sdf(latent_dir,meshdir)

from postprocess import postprocess
for i in range(len(views)):
    for j in range(len(views[0])):
        t  = np.load(os.path.join(layoutdir,'{}_{}.npy'.format(i,j))).astype(np.float32)
        if t.max() == 0:
            try:
                os.remove(os.path.join(meshdir,'{}_{}.ply'.format(i,j)))
            except:
                continue
postprocess(outputdir)
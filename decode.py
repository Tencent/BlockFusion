import random
import torch
import torch.utils.data
from glob import glob
import os
import json, csv
import time
from tqdm.auto import tqdm
import numpy as np
from autoencoder import VAE
import mesh
# from utils.reconstruct import *
# from diff_utils.helpers import *
from decoder import SdfModel
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from postprocess import postprocess
@torch.no_grad()
def test_vae_sdf(latent_dir,output_dir):

    vae_config = {"kl_std": 0.25,
                  "kl_weight": 0.001,
                  "plane_shape": [3, 32, 128, 128],
                  "z_shape": [2, 32, 32],
                  "num_heads": 16
                  }
    decoder_config = {
        "skip_in": [],
        "n_layers": 3,
        "width": 128,
        "channels": 32,
        "ckpt_path": 'checkpoints/mlp.tar'
    }
    decoder_model = SdfModel(decoder_config)

    model = VAE(vae_config).cuda()

    resume_params_path = "checkpoints/vae/epoch=68.ckpt"
    state_dict = torch.load(resume_params_path,map_location='cpu')["state_dict"]
    vae_state_dict= {}
    for k,v in state_dict.items():
        if k.startswith("vae_model."):
            vae_state_dict[k[10:]] = state_dict[k]
    model.load_state_dict(vae_state_dict)
    model = model.cuda().eval()

    print("load only state_dict from checkpoint: {}".format(resume_params_path))
    latents = glob(os.path.join(latent_dir, '*npy'))
    random.shuffle(latents)
    stats_dir = 'checkpoints/vae_stats'
    min_values = np.load(f'{stats_dir}/lower_bound.npy').astype(np.float32).reshape(-1, 1, 1)  # should be (1, 96, 1, 1)
    max_values = np.load(f'{stats_dir}/upper_bound.npy').astype(np.float32).reshape(-1, 1, 1)
    _range = (max_values - min_values)
    middle = ((min_values + max_values) / 2)

    middle = torch.from_numpy(middle).cuda()
    _range = torch.from_numpy(_range).cuda()
    with tqdm(latents) as pbar:
        for idx, data in enumerate(pbar):
            pbar.set_description("Files evaluated: {}/{}".format(idx, len(latents)))
            mesh_name = data.split('/')[-1].split('.')[0]
            z = np.load(data)
            z = torch.from_numpy(z).cuda()
            recon = model.decode(z)
            recon = recon.view(-1, 96, 128, 128) * (_range / 2) + middle
            recon = recon.view(-1, 3, 32, 128, 128)
            mesh_filename = os.path.join(output_dir,mesh_name)
            mesh.create_mesh(decoder_model, recon, mesh_filename, N=256, max_batch=2**21, from_plane_features=True)

# debug for vae

# if __name__ == "__main__":
#     import argparse
#
#     arg_parser = argparse.ArgumentParser()
#     arg_parser.add_argument(
#         "--exp_dir", "-e", default='config/stage1_vae_sdf_wu_v3',
#         help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
#     )
#     arg_parser.add_argument(
#         "--resume_params", default='68',
#         help="continue from previous saved logs, integer value, 'last', or 'finetune'",
#     )
#
#     arg_parser.add_argument("--num_samples", "-n", default=20000, type=int,
#                             help='number of samples to generate and reconstruct')
#
#     arg_parser.add_argument("--filter", default=False, help='whether to filter when sampling conditionally')
#
#     args = arg_parser.parse_args()
#     # specs = json.load(open(os.path.join(args.exp_dir, "specs_test.json")))
#     # print(specs["Description"])
#
#     # latent_dir = os.path.join(args.exp_dir, "modulations")
#     # os.makedirs(latent_dir, exist_ok=True)
#
#     recon_dir = os.path.join(args.exp_dir, "recon" + time.strftime('%Y-%m-%d-%H:%M:%S'))
#     os.makedirs(recon_dir, exist_ok=True)
#     test_vae_sdf()
#
#     postprocess(target=os.path.join(recon_dir,'0_0', "reconstruct.ply"),source=os.path.join(recon_dir,'0_1', "reconstruct.ply"),output=os.path.join(recon_dir, "postprocessed.ply"))



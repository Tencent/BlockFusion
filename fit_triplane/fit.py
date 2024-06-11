from tqdm import tqdm
from easydict import EasyDict as edict
import argparse
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import open3d as o3d
import mcubes, trimesh
import torch
import numpy as np
import os
import json

def vis_model(net, triplane, n_labels, savedir, oid=0, rank=0):
    os.makedirs(savedir, exist_ok=True)
    for pid in range(n_labels):
        plot_shape(net, triplane, triplane.R * 2, n_labels, 0.0, os.path.join(savedir, f"triplane.ply"),pid, oid)


def save_model(net, triplane, savedir, rank=0):
    os.makedirs(savedir, exist_ok=True)
    torch.save(triplane.state_dict(), os.path.join(savedir, f"tripalne.tar"))


def create_optimizer(net, triplane, config):
    params_to_train = []
    if net is not None:
        params_to_train += [{'name':'net', 'params':net.parameters(), 'lr':config.lr_net}]
    if triplane is not None:
        params_to_train += [{'name':'tri', 'params':triplane.parameters(), 'lr':config.lr_tri}]
    return torch.optim.Adam(params_to_train)

def update_lr(optimizer, epoch, config):
    learning_factor = (np.cos(np.pi * epoch / config.max_iters) + 1.0) * 0.5 * (1 - 0.001) + 0.001
    for param_group in optimizer.param_groups:
        if "net" in param_group['name']:
            param_group['lr'] = config.lr_net * learning_factor
        if "tri" in param_group['name']:
            param_group['lr'] = config.lr_tri * learning_factor

def extract_fields(bound_min, bound_max, resolution, query_func, channel):
    N = 128 # 64. Change it when memory is insufficient!
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution, channel], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs), channel).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def plot_shape(net, triplane, resolution, channel, threshold, savedir, pid, oid):
    u = extract_fields(
        bound_min=[-1.0, -1.0, -1.0],
        bound_max=[ 1.0,  1.0,  1.0],
        resolution=resolution,
        query_func=lambda xyz: -net(triplane(xyz, oid)),
        channel=channel,
    )
    if pid<0:
        u = np.max(u, -1)  # sdf of scene
    else:
        u = u[..., pid]  # sdf of part
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    vertices = vertices / (resolution - 1.0) * 2 - 1
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.export(savedir)

def get_triangle_points(obj):
    obj.compute_triangle_normals()
    vertices = np.asarray(obj.vertices)
    triangles = np.asarray(obj.triangles)
    normals = np.asarray(obj.triangle_normals)

    tri_points = torch.from_numpy(vertices[triangles].mean(1)).float()
    tri_normals = torch.from_numpy(normals).float()
    return tri_points, tri_normals


class CropDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.datadir = config.objpaths
        cropid = int(self.datadir.split("/")[-1][4:-4])  # xx/xx/crop{}.obj
        self.manifold = o3d.io.read_triangle_mesh(self.datadir)
        self.tri_points, self.tri_normals = get_triangle_points(self.manifold)
        self.pcd_num = config.pcd_num
        assert self.pcd_num > len(self.tri_points)

        sdf_npz = np.load(os.path.join(os.path.dirname(self.datadir), "sdf_points.npz"))
        self.bbox = torch.from_numpy(
            np.load(os.path.join(os.path.dirname(self.datadir), "bbxs.npy"))[cropid]
        ).float()  # [2,3]
        self.scale, self.translate = self.parse_bbox(self.bbox)
        self.sdf_points, self.sdf_sdfs = self.crop_sdf_gt(sdf_npz, self.bbox)  # [Ns,3], [Ns,1]
        self.sdf_num = len(self.sdf_sdfs)
        self.points = None
        self.normals = None
        self.masks = None
        self.sdfs = None

        self.N = self.pcd_num + self.sdf_num
        self.B = config.batch_size
        self.perm()

        self.length = (self.N - 1) // self.B + 1
        self.i = 0

    def parse_bbox(self, bbox):
        aa, bb = bbox
        return 0.625, -(aa + bb) / 2


    def crop_sdf_gt(self, sdf_npz, bbox):
        pts = torch.from_numpy(sdf_npz['query_points']).float()  # [N,3]
        sdf = torch.from_numpy(sdf_npz['sdf']).float()  # [N,]
        aa, bb = bbox
        maskx = torch.logical_and(pts[:, 0] > aa[0], pts[:, 0] < bb[0])
        masky = torch.logical_and(pts[:, 1] > aa[1], pts[:, 1] < bb[1])
        maskz = torch.logical_and(pts[:, 2] > aa[2], pts[:, 2] < bb[2])
        mask = torch.logical_and(maskx, torch.logical_and(masky, maskz))
        return pts[mask], sdf[mask].unsqueeze(-1)

    def sample_points(self, ):
        pcd = self.manifold.sample_points_uniformly(self.pcd_num - len(self.tri_points), True)

        pcd_points = np.asarray(pcd.points)  # [N,3]
        pcd_normals = np.asarray(pcd.normals)  # [N,3]
        # pcd_colors = np.asarray(pcd.colors)  # [N,4]
        return pcd_points, pcd_normals

    def perm(self):
        pcd_points, pcd_normals = self.sample_points()
        pcd_points = torch.from_numpy(pcd_points).float()
        pcd_normals = torch.from_numpy(pcd_normals).float()

        # [N,3], [N,3], [N,1], [N,]
        self.points = torch.cat([pcd_points, self.tri_points, self.sdf_points])
        self.normals = torch.cat([pcd_normals, self.tri_normals, torch.zeros_like(self.sdf_points)])
        self.sdfs = torch.cat([torch.zeros(self.pcd_num, 1), self.sdf_sdfs])
        self.masks = torch.cat([torch.ones(self.pcd_num), torch.zeros(self.sdf_num)]).bool()

        perm = torch.from_numpy(np.random.permutation(self.N))
        self.points = self.points[perm]
        self.normals = self.normals[perm]
        self.sdfs = self.sdfs[perm]
        self.masks = self.masks[perm]

        self.points = (self.points + self.translate) * self.scale

    def __getitem__(self, oid):
        if self.i + 1 < self.length:
            points = self.points[self.i * self.B:(self.i + 1) * self.B].cuda()
            normals = self.normals[self.i * self.B:(self.i + 1) * self.B].cuda()
            sdfs = self.sdfs[self.i * self.B:(self.i + 1) * self.B].cuda()
            masks = self.masks[self.i * self.B:(self.i + 1) * self.B].cuda()
            self.i += 1
        else:
            points = self.points[self.i * self.B:].cuda()
            normals = self.normals[self.i * self.B:].cuda()
            sdfs = self.sdfs[self.i * self.B:].cuda()
            masks = self.masks[self.i * self.B:].cuda()
            self.i = 0
            # self.perm()
        return points, normals, sdfs, masks


class Triplane(nn.Module):
    def __init__(self,
                 n=1,
                 reso=256,
                 channel=32,
                 init_type="geo_init",
                 objname=None,
                 ):
        super().__init__()
        self.n = n
        self.objname = objname
        # assert len(self.objname) == n
        if init_type == "geo_init":
            sdf_proxy = nn.Sequential(
                nn.Linear(3, channel), nn.Softplus(beta=100),
                nn.Linear(channel, channel),
            )
            torch.nn.init.constant_(sdf_proxy[0].bias, 0.0)
            torch.nn.init.normal_(sdf_proxy[0].weight, 0.0, np.sqrt(2) / np.sqrt(channel))
            torch.nn.init.constant_(sdf_proxy[2].bias, 0.0)
            torch.nn.init.normal_(sdf_proxy[2].weight, 0.0, np.sqrt(2) / np.sqrt(channel))

            ini_sdf = torch.zeros([3, channel, reso, reso])
            X = torch.linspace(-1.0, 1.0, reso)
            (U, V) = torch.meshgrid(X, X, indexing="ij")
            Z = torch.zeros(reso, reso)
            inputx = torch.stack([Z, U, V], -1).reshape(-1, 3)
            inputy = torch.stack([U, Z, V], -1).reshape(-1, 3)
            inputz = torch.stack([U, V, Z], -1).reshape(-1, 3)
            ini_sdf[0] = sdf_proxy(inputx).permute(1, 0).reshape(channel, reso, reso)
            ini_sdf[1] = sdf_proxy(inputy).permute(1, 0).reshape(channel, reso, reso)
            ini_sdf[2] = sdf_proxy(inputz).permute(1, 0).reshape(channel, reso, reso)

            self.triplane = torch.nn.Parameter(ini_sdf.unsqueeze(0).repeat(self.n, 1, 1, 1, 1) / 3, requires_grad=True)
        elif init_type == "zero_init":
            self.triplane = torch.nn.Parameter(torch.zeros([self.n, 3, channel, reso, reso]), requires_grad=True)

        self.R = reso
        self.C = channel
        self.register_buffer("plane_axes", torch.tensor(
            [[[0, 1, 0],
              [1, 0, 0],
              [0, 0, 1]],
             [[0, 0, 1],
              [1, 0, 0],
              [0, 1, 0]],
             [[0, 1, 0],
              [0, 0, 1],
              [1, 0, 0]]], dtype=torch.float32)
                             )

        # xy xz yz

    def project_onto_planes(self, xyz):
        M, _ = xyz.shape
        xyz = xyz.unsqueeze(0).expand(3, -1, -1).reshape(3, M, 3)
        inv_planes = torch.linalg.inv(self.plane_axes).reshape(3, 3, 3)
        projections = torch.bmm(xyz, inv_planes)
        return projections[..., :2]  # [3, M, 2]

    def forward(self, xyz, oid):
        # pts: [M,3]
        M, _ = xyz.shape
        plane_features = self.triplane[oid:oid + 1].view(3, self.C, self.R, self.R)
        projected_coordinates = self.project_onto_planes(xyz).unsqueeze(1)
        feats = F.grid_sample(
            plane_features,  # [3,C,R,R]
            projected_coordinates.float(),  # [3,1,M,2]
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )  # [3,C,1,M]
        feats = feats.permute(0, 3, 2, 1).reshape(3, M, self.C).sum(0)
        return feats  # [M,C]

    def update_resolution(self, new_reso):
        old_tri = self.triplane.data.view(self.n * 3, self.C, self.R, self.R)
        new_tri = F.interpolate(old_tri, size=(new_reso, new_reso), mode='bilinear', align_corners=True)
        self.R = new_reso
        self.triplane = torch.nn.Parameter(new_tri.view(self.n, 3, self.C, self.R, self.R), requires_grad=True)


class Network(nn.Module):
    def __init__(self,
                 d_in=32,
                 d_hid=128,
                 n_layers=3,
                 d_out=6,
                 init_type="geo_init",
                 weight_norm=True,
                 bias=0.5,
                 inside_outside=False
                 ):
        super().__init__()
        dims = [d_in] + [d_hid for _ in range(n_layers)] + [d_out]
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            in_dim = dims[l]
            out_dim = dims[l + 1]
            lin = nn.Linear(in_dim, out_dim)

            if init_type == "geo_init":
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, feats):
        x = feats
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        return x


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='base.json')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

torch.cuda.set_device(f"cuda:{args.gpu}")
with open(args.config, 'r') as f:
    config = json.load(f)
config = edict(config)
assert len(config.fixmlp) > 0

net = Network(
    d_in=config.channel,
    d_hid=config.n_hid,
    n_layers=config.n_layers,
    d_out=config.n_labels,
    init_type="geo_init",
).cuda()
net.load_state_dict(torch.load(config.fixmlp, map_location='cuda'))

dataset = CropDataset(
    config
)

triplane = Triplane(
    reso=config.resolution // (2 ** len(config.c2f_scale)),
    channel=config.channel,
    init_type="geo_init",
    objname=None,
).cuda()

optimizer = create_optimizer(None, triplane, config)

max_len = dataset.length

eps_s = 1e-6
eps_v = 1e-6
w_eik = config.w_eik
w_nor = config.w_nor
w_sur = config.w_sur
w_sdf = config.w_sdf
for epoch in tqdm(range(1, config.max_iters + 1)):
    eik_loss_list = []
    nor_loss_list = []
    sur_loss_list = []
    sdf_loss_list = []
    loss_list = []

    if epoch % 20 == 0:
        dataset.perm()

    if epoch in config.c2f_scale:
        new_reso = int(config.resolution / (2 ** (len(config.c2f_scale) - config.c2f_scale.index(epoch) - 1)))
        triplane.update_resolution(new_reso)
        optimizer = create_optimizer(None, triplane, config)
        update_lr(optimizer, epoch - 1, config)
        torch.cuda.empty_cache()

    for _ in range(max_len):
        points, normals, sdfs, masks = dataset[0]
        gt_normals = normals[masks]
        gt_sdfs = sdfs[~masks]

        mnfd_points = points[masks]
        unif_points = points[~masks]
        len_mnfd = mnfd_points.shape[0]
        len_unif = unif_points.shape[0]
        bkpt = [len_mnfd * k for k in range(6)] + [len_mnfd * 6 + len_unif * k for k in range(7)]
        rndm_points = torch.from_numpy(np.random.uniform(-1., 1., size=(len_unif, 3)).astype(np.float32)).cuda()

        points_all = torch.cat([
            mnfd_points + torch.as_tensor([[eps_s, 0.0, 0.0]]).to(points),
            mnfd_points + torch.as_tensor([[-eps_s, 0.0, 0.0]]).to(points),
            mnfd_points + torch.as_tensor([[0.0, eps_s, 0.0]]).to(points),
            mnfd_points + torch.as_tensor([[0.0, -eps_s, 0.0]]).to(points),
            mnfd_points + torch.as_tensor([[0.0, 0.0, eps_s]]).to(points),
            mnfd_points + torch.as_tensor([[0.0, 0.0, -eps_s]]).to(points),
            rndm_points + torch.as_tensor([[eps_v, 0.0, 0.0]]).to(points),
            rndm_points + torch.as_tensor([[-eps_v, 0.0, 0.0]]).to(points),
            rndm_points + torch.as_tensor([[0.0, eps_v, 0.0]]).to(points),
            rndm_points + torch.as_tensor([[0.0, -eps_v, 0.0]]).to(points),
            rndm_points + torch.as_tensor([[0.0, 0.0, eps_v]]).to(points),
            rndm_points + torch.as_tensor([[0.0, 0.0, -eps_v]]).to(points),
            points,
        ], dim=0)  # [N,3]

        sdfs_all = net(triplane(points_all, 0))  # [N*, L]
        pred_sdfs = sdfs_all[bkpt[12]:]  # [N, L]
        mnfd_grad = torch.cat([
            0.5 * (sdfs_all[bkpt[0]:bkpt[1]] - sdfs_all[bkpt[1]:bkpt[2]]) / eps_s,
            0.5 * (sdfs_all[bkpt[2]:bkpt[3]] - sdfs_all[bkpt[3]:bkpt[4]]) / eps_s,
            0.5 * (sdfs_all[bkpt[4]:bkpt[5]] - sdfs_all[bkpt[5]:bkpt[6]]) / eps_s,
        ], dim=-1)
        rndm_grad = torch.stack([
            0.5 * (sdfs_all[bkpt[6]:bkpt[7]] - sdfs_all[bkpt[7]:bkpt[8]]) / eps_v,
            0.5 * (sdfs_all[bkpt[8]:bkpt[9]] - sdfs_all[bkpt[9]:bkpt[10]]) / eps_v,
            0.5 * (sdfs_all[bkpt[10]:bkpt[11]] - sdfs_all[bkpt[11]:bkpt[12]]) / eps_v,
        ], dim=-1)

        # loss
        eik_loss = ((rndm_grad.norm(2, dim=-1) - 1) ** 2).mean()
        nor_loss = (mnfd_grad - gt_normals).norm(2, dim=1).mean()
        sur_loss = pred_sdfs[masks].abs().mean()
        sdf_loss = (pred_sdfs[~masks] - gt_sdfs).abs().mean()

        loss = eik_loss * w_eik + nor_loss * w_nor + sur_loss * w_sur + sdf_loss * w_sdf
        eik_loss_list.append(eik_loss.item())
        nor_loss_list.append(nor_loss.item())
        sur_loss_list.append(sur_loss.item())
        sdf_loss_list.append(sdf_loss.item())
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % config.i_print == 0:
        print(
            f"L={np.mean(loss_list):.4f}, Leik={np.mean(eik_loss_list):.4f}, Lnor={np.mean(nor_loss_list):.4f}, Lsur={np.mean(sur_loss_list):.4f}, Lsdf={np.mean(sdf_loss_list):.4f}")

    update_lr(optimizer, epoch, config)

vis_model(net, triplane, config.n_labels, '.')
save_model(net, triplane, '.')


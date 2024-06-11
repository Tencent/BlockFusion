import torch.nn as nn
BCE = nn.BCELoss()
import open3d as o3d
import torch.optim as optim
from easydict import EasyDict as edict
import argparse

from typing import Union
import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
from torch.autograd.functional import jacobian
import numpy as np
import os
from math import ceil
import glob
import shutil
def _validate_chamfer_reduction_inputs(
        batch_reduction: Union[str, None], point_reduction: str
):
    """Check the requested reductions are valid.
    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
        points: Union[torch.Tensor, Pointclouds],
        lengths: Union[torch.Tensor, None],
        normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
                lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


def compute_truncated_chamfer_distance(
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
        trunc=0.2,
        batch_reduction: Union[str, None] = "mean",
        point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    # truncation
    x_mask[cham_x >= trunc] = True
    y_mask[cham_y >= trunc] = True
    cham_x[x_mask] = 0.0
    cham_y[y_mask] = 0.0

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # Apply point reduction

    # cham_x = cham_x.sum(1)  # (N,)
    # cham_y = cham_y.sum(1)  # (N,)

    # use l1 norm, more robust to partial case
    cham_x = torch.sqrt(cham_x).sum(1)  # (N,)
    cham_y = torch.sqrt(cham_y).sum(1)  # (N,)

    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    # cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist


def arap_cost(R, t, g, e, w, lietorch=True):
    '''
    :param R:
    :param t:
    :param g:
    :param e:
    :param w:
    :return:
    '''

    R_i = R[:, None]
    g_i = g[:, None]
    t_i = t[:, None]

    g_j = g[e]
    t_j = t[e]

    # if lietorch :
    #     e_ij = ((R_i * (g_j - g_i) + g_i + t_i  - g_j - t_j )**2).sum(dim=-1)

    e_ij = (((R_i @ (g_j - g_i)[..., None]).squeeze() + g_i + t_i - g_j - t_j) ** 2).sum(dim=-1)

    o = (w * e_ij).mean()

    return o


def silhouette_cost(x, y, renderer):
    INF = 1e+6

    px, dx = renderer(x)
    py, dy = renderer(y)

    px, dx, py, dy = map(lambda feat: feat.squeeze(), [px, dx, py, dy])

    dx[dx < 0] = INF
    dy[dy < 0] = INF

    dx, _ = torch.min(dx, dim=-1, )
    dy, _ = torch.min(dy, dim=-1)

    dx[dx == INF] = 0
    dy[dy == INF] = 0

    x_mask = px[..., 0] > 0
    y_mask = py[..., 0] > 0

    # plt.imshow(py.detach().cpu().numpy())
    # plt.show()
    #
    # # plt.figure(figsize=(10, 10))
    # plt.imshow(dx.detach().cpu().numpy())
    # plt.show()
    # # #
    # plt.imshow(dy.detach().cpu().numpy())
    # plt.show()

    depth_error = (dx - dy) ** 2
    #
    # plt.imshow(depth_error.detach().cpu().numpy())
    # plt.show()

    # depth_error[depth_error>0.01] = 0

    silh_error = (px - py) ** 2

    silh_error = silh_error[~y_mask]

    depth_error = depth_error[y_mask * x_mask]

    depth_error[depth_error > 0.06 ** 2] = 0
    # plt.imshow(depth_error.detach().cpu().numpy())
    # plt.show()

    silh_loss = torch.mean(silh_error)
    depth_loss = torch.mean(depth_error)

    return silh_loss + depth_loss




def chamfer_dist(src_pcd, tgt_pcd, samples=1000):
    '''
    :param src_pcd: warpped_pcd
    :param R: node_rotations
    :param t: node_translations
    :param data:
    :return:
    '''

    """chamfer distance"""
    src = torch.randperm(src_pcd.shape[0])
    tgt = torch.randperm(tgt_pcd.shape[0])
    s_sample = src_pcd[src[:samples]]
    t_sample = tgt_pcd[tgt[:samples]]
    cham_dist = compute_truncated_chamfer_distance(s_sample[None], t_sample[None], trunc=1e+10)

    return cham_dist


def nerfies_regularization(jacobian, eps=1e-6):
    jacobian = jacobian.cpu().double()
    svals = jacobian.svd(compute_uv=False).S  # small SVD runs faster on cpu
    svals[svals < eps] = eps
    log_svals = torch.log(svals.max(dim=1)[0])
    loss = torch.mean(log_svals ** 2)
    return loss


def scene_flow_metrics(pred, labels, strict=0.025, relax=0.05):
    l2_norm = torch.sqrt(torch.sum((pred - labels) ** 2, 1)).cpu()  # Absolute distance error.
    labels_norm = torch.sqrt(torch.sum(labels * labels, 1)).cpu()
    relative_err = l2_norm / (labels_norm + 1e-20)

    EPE3D = torch.mean(l2_norm).item()  # Mean absolute distance error

    # NOTE: AccS
    error_lt_5 = torch.BoolTensor((l2_norm < strict))
    relative_err_lt_5 = torch.BoolTensor((relative_err < strict))
    AccS = torch.mean((error_lt_5 | relative_err_lt_5).float()).item()

    # NOTE: AccR
    error_lt_10 = torch.BoolTensor((l2_norm < relax))
    relative_err_lt_10 = torch.BoolTensor((relative_err < relax))
    AccR = torch.mean((error_lt_10 | relative_err_lt_10).float()).item()

    # NOTE: outliers
    relative_err_lt_30 = torch.BoolTensor(relative_err > 0.3)
    outlier = torch.mean(relative_err_lt_30.float()).item()

    return EPE3D * 100, AccS * 100, AccR * 100, outlier * 100


def scene_flow_EPE_np(pred, labels, mask):
    '''
    :param pred: [B, N, 3]
    :param labels:
    :param mask: [B, N]
    :return:
    '''
    error = np.sqrt(np.sum((pred - labels) ** 2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels * labels, 2) + 1e-20)  # B,N
    acc1 = np.sum(np.logical_or((error <= 0.05), (error / gtflow_len <= 0.05)), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1), (error / gtflow_len <= 0.1)), axis=1)

    mask_sum = np.sum(mask, 1)
    acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = np.sum(acc1)
    acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = np.sum(acc2)

    EPE = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    EPE = np.sum(EPE)
    return EPE, acc1, acc2


def compute_flow_metrics(flow, flow_gt, overlap=None):
    metric_info = {}

    # full point cloud
    epe, AccS, AccR, outlier = scene_flow_metrics(flow, flow_gt)
    metric_info.update(
        {
            "full-epe": epe,
            "full-AccS": AccS,
            "full-AccR": AccR,
            "full-outlier": outlier
        }
    )

    if overlap is not None:
        # visible
        epe, AccS, AccR, outlier = scene_flow_metrics(flow[overlap], flow_gt[overlap])
        metric_info.update(
            {
                "vis-epe": epe,
                "vis-AccS": AccS,
                "vis-AccR": AccR,
                "vis-outlier": outlier

            }
        )

        # occluded
        epe, AccS, AccR, outlier = scene_flow_metrics(flow[~overlap], flow_gt[~overlap])
        metric_info.update(
            {
                "occ-epe": epe,
                "occ-AccS": AccS,
                "occ-AccR": AccR,
                "occ-outlier": outlier
            }
        )

    return metric_info



def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




def _6d_to_SO3(d6):
    '''
    On the Continuity of Rotation Representations in Neural Networks, CVPR'19. c.f. http://arxiv.org/abs/1812.07035
    :param d6: [n, 6]
    :return: [n, 3, 3]
    '''
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def euler_to_SO3(euler_angles, convention = ['X', 'Y', 'Z']):
    '''
    :param euler_angles: [n, 6]
    :param convention: order of axis
    :return:
    '''

    def _axis_angle_rotation(axis, angle):
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)
        if axis == "X":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "Y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "Z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        else:
            raise ValueError("letter must be either X, Y or Z.")
        return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]

    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def quaternion_to_SO3(quaternions):
    '''
    :param quaternions: [n, 4]
    :return:
    '''

    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))



def skew(w):

    zero = torch.zeros_like( w[...,0])
    W = torch.stack ( [ zero, -w[...,2], w[...,1],
                        w[...,2], zero, -w[...,0],
                        -w[...,1], w[...,0], zero], dim=-1).reshape((-1,3,3))
    return W

def exp_se3( w, v, theta):
    '''
    :param w:
    :param v:
    :param theta:
    :return:
    '''

    theta=theta[...,None]
    W = skew(w)
    I= torch.eye(3)[None].to(theta)
    R = I + torch.sin(theta) * W +  (1-torch.cos(theta)) * W @ W
    p = I + (1-torch.cos(theta) ) * W + (theta - torch.sin(theta)) * W @ W
    t = p @ v[...,None]
    return R, t

def exp_so3( w, theta):

    theta=theta[...,None]
    W = skew(w)
    I= torch.eye(3)[None].to(theta)
    R = I + torch.sin(theta) * W +  (1-torch.cos(theta)) * W @ W
    return R



def get_views(panorama_height, panorama_width, window_size=64, stride=32):
    # panorama_height /= 8
    # panorama_width /= 8
    num_blocks_height = ceil((panorama_height - window_size) / stride) + 1
    num_blocks_width = ceil((panorama_width - window_size) / stride) + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = [[] for _ in range(num_blocks_height)]
    ids = [[] for _ in range(num_blocks_height)]
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
    # for i in range(total_num_blocks):
            h_start = int(i  * stride)
            h_end = h_start + window_size
            w_start = int(j  * stride)
            w_end = w_start + window_size
            views[i].append((h_start, h_end, w_start, w_end))
            ids[i].append((i,j))
    return views,ids


class Deformation_Pyramid ():

    def __init__(self, depth, width, device, k0, m, rotation_format, nonrigidity_est=False, motion='SE3'):

        pyramid = []


        assert motion in [ "Sim3", "SE3", "sflow"]


        for i in range (m):
            pyramid.append(
                NDPLayer(depth,
                         width,
                         k0,
                         i+1,
                         rotation_format,
                         nonrigidity_est=nonrigidity_est & (i!=0),
                         motion=motion
                         ).to(device)
            )


        self.pyramid = pyramid
        self.n_hierarchy = m

    def warp(self, x, max_level=None, min_level=0):

        if max_level is None:
            max_level = self.n_hierarchy - 1

        assert max_level < self.n_hierarchy, "more level than defined"

        data = {}

        for i in range(min_level, max_level + 1):
            x, nonrigidity = self.pyramid[i](x)
            data[i] = (x, nonrigidity)
        return x, data

    def gradient_setup(self, optimized_level):

        assert optimized_level < self.n_hierarchy, "more level than defined"

        # optimize current level, freeze the other levels
        for i in range( self.n_hierarchy):
            net = self.pyramid[i]
            if i == optimized_level:
                for param in net.parameters():
                    param.requires_grad = True
            else:
                for param in net.parameters():
                    param.requires_grad = False



class NDPLayer(nn.Module):
    def __init__(self, depth, width, k0, m, rotation_format="euler", nonrigidity_est=False, motion='SE3'):
        super().__init__()

        self.k0 = k0
        self.m = m
        dim_x =  6
        self.nonrigidity_est = nonrigidity_est
        self.motion = motion
        self.input= nn.Sequential( nn.Linear(dim_x,width), nn.ReLU())
        self.mlp = MLP(depth=depth,width=width)

        self.rotation_format = rotation_format


        """rotation branch"""
        if self.motion in [ "Sim3", "SE3"] :

            if self.rotation_format in [ "axis_angle", "euler" ]:
                self.rot_brach = nn.Linear(width, 3)
            elif self.rotation_format == "quaternion":
                self.rot_brach = nn.Linear(width, 4)
            elif self.rotation_format == "6D":
                self.rot_brach = nn.Linear(width, 6)


            if self.motion == "Sim3":
                self.s_branch = nn.Linear(width, 1) # scale branch


        """translation branch"""
        self.trn_branch = nn.Linear(width, 3)


        """rigidity branch"""
        if self.nonrigidity_est:
            self.nr_branch = nn.Linear(width, 1)
            self.sigmoid = nn.Sigmoid()


        # Apply small scaling on the MLP output, s.t. the optimization can start from near identity pose
        self.mlp_scale = 0.001

        self._reset_parameters()

    def forward (self, x):

        fea = self.posenc( x )
        fea = self.input(fea)
        fea = self.mlp(fea)

        t = self.mlp_scale * self.trn_branch ( fea )

        if self.motion == "SE3":
            R = self.get_Rotation(fea)
            x_ = (R @ x[..., None]).squeeze() + t

        elif self.motion == "Sim3":
            R = self.get_Rotation(fea)
            s = self.mlp_scale * self.s_branch(fea) + 1  # optimization starts with scale==1
            x_ = s * (R @ x[..., None]).squeeze() + t

        else: # scene flow
            x_ = x + t


        if self.nonrigidity_est:
            nonrigidity =self.sigmoid( self.mlp_scale * self.nr_branch(fea) )
            x_ = x + nonrigidity * (x_ - x)
            nonrigidity = nonrigidity.squeeze()
        else:
            nonrigidity = None


        return x_.squeeze(), nonrigidity



    def get_Rotation (self, fea):

        R = self.mlp_scale * self.rot_brach( fea )

        if self.rotation_format == "euler":
            R = euler_to_SO3(R)
        elif self.rotation_format == "axis_angle":
            theta = torch.norm(R, dim=-1, keepdim=True)
            w = R / theta
            R = exp_so3(w, theta)
        elif self.rotation_format =='quaternion':
            s = (R * R).sum(1)
            R = R / _copysign(torch.sqrt(s), R[:, 0])[:, None]
            R = quaternion_to_SO3(R)
        elif self.rotation_format == "6D":
            R = _6d_to_SO3(R)

        return R


    def posenc(self, pos):
        pi = 3.14
        x_position, y_position, z_position = pos[..., 0:1], pos[..., 1:2], pos[..., 2:3]
        # mul_term = ( 2 ** (torch.arange(self.m, device=pos.device).float() + self.k0) * pi ).reshape(1, -1)
        mul_term = (2 ** (self.m + self.k0)  )#.reshape(1, -1)

        sinx = torch.sin(x_position * mul_term)
        cosx = torch.cos(x_position * mul_term)
        siny = torch.sin(y_position * mul_term)
        cosy = torch.cos(y_position * mul_term)
        sinz = torch.sin(z_position * mul_term)
        cosz = torch.cos(z_position * mul_term)
        pe = torch.cat([sinx, cosx, siny, cosy, sinz, cosz], dim=-1)
        return pe


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)




class MLP(torch.nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        self.pts_linears = nn.ModuleList( [nn.Linear(width, width) for i in range(depth - 1)])

    def forward(self, x):
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
            x = F.relu(x)
        return x

class Deformer2():

    def __init__(self, config):
        self.config = config
        if self.config.gpu_mode:
            self.config.device = torch.cuda.current_device()
        else:
            self.config.device = torch.device('cpu')


        # if landmarks is not None:
        #     s_ldmk , t_ldmk = landmarks
        #     self.landmarks = (s_ldmk, t_ldmk)
        # else:
        #     self.landmarks = None


    def train_field(self, src_pcd, tgt_pcd, landmarks=None, cancel_translation=False):
        """
        Args:
            src_pcd: [n,3] numpy array
            tgt_pcd: [n,3] numpy array
        Returns:
        """

        config = self.config



        src_pcd, tgt_pcd = map(lambda x: torch.from_numpy(x).to(config.device), [src_pcd, tgt_pcd])


        """construct model"""
        NDP = Deformation_Pyramid(depth=config.depth,
                                  width=config.width,
                                  device=config.device,
                                  k0=config.k0,
                                  m=config.m,
                                  nonrigidity_est=config.w_reg > 0,
                                  rotation_format=config.rotation_format,
                                  motion=config.motion_type)



        """cancel global translation"""
        if cancel_translation:
            src_mean = src_pcd.mean(dim=0, keepdims=True)
            tgt_mean = tgt_pcd.mean(dim=0, keepdims=True)
        else:
            src_mean = 0.
            tgt_mean = 0.


        src_pcd = src_pcd - src_mean
        tgt_pcd = tgt_pcd - tgt_mean

        self.src_mean = src_mean
        self.tgt_mean = tgt_mean


        s_sample = src_pcd
        t_sample = tgt_pcd


        if landmarks is not None:

            src_ldmk = torch.from_numpy(landmarks[0]).to(config.device) - src_mean
            tgt_ldmk = torch.from_numpy(landmarks[1]).to(config.device) - tgt_mean



        for level in range(NDP.n_hierarchy):

            """freeze non-optimized level"""
            NDP.gradient_setup(optimized_level=level)

            optimizer = optim.Adam(NDP.pyramid[level].parameters(), lr=config.lr)

            break_counter = 0
            loss_prev = 1e+6


            # inpcd = torch.concatenate( [s_sample, x0_plane] , dim=0 )

            """optimize current level"""
            for iter in range(config.iters):

                # s_sample_warped, data = NDP.warp(s_sample, max_level=level, min_level=level)
                # inpcd_warped, data = NDP.warp(inpcd, max_level=level, min_level=level)

                if landmarks is not None:

                    src_pts = torch.cat([src_ldmk, s_sample])
                    warped_pts, data = NDP.warp(src_pts, max_level=level, min_level=level)
                    warped_ldmk = warped_pts[: len(src_ldmk)]
                    s_sample_warped = warped_pts[len(src_ldmk):]
                    loss_ldmk = torch.mean(torch.sum((warped_ldmk - tgt_ldmk) ** 2, dim=-1))
                    loss_chamfer = compute_truncated_chamfer_distance(s_sample_warped[None], t_sample[None])

                    loss = config.w_ldmk*loss_ldmk + config.w_cd * loss_chamfer
                else :
                    s_sample_warped, data = NDP.warp(s_sample, max_level=level, min_level=level)
                    # chamfer distance
                    loss_chamfer = compute_truncated_chamfer_distance(s_sample_warped[None], t_sample[None], trunc=1e+9)
                    loss_ldmk = torch.tensor(0)
                    loss = loss_chamfer

                print(
                    "level-", level ,
                    f"L={loss.item():.4f}, "
                    f"Lcham={loss_chamfer.item():.4f}, "
                    f"Lldmk={loss_ldmk.item():.4f}"
                )

                if level > 0 and config.w_reg > 0:
                    nonrigidity = data[level][1]
                    target = torch.zeros_like(nonrigidity)
                    reg_loss = BCE(nonrigidity, target)
                    loss = loss + config.w_reg * reg_loss

                # early stop
                if loss.item() < 1e-4:
                    break
                if abs(loss_prev - loss.item()) < loss_prev * config.break_threshold_ratio:
                    break_counter += 1
                if break_counter >= config.max_break_count:
                    break
                loss_prev = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # use warped points for next level
            # use warped points for next level
            if landmarks is not None:
                src_ldmk = warped_ldmk.detach()

            s_sample = s_sample_warped.detach()

        self.NDP = NDP

    def warp_points(self, points):
        '''
        Args:
            points: [n,3] numpy array
        Returns:

        '''

        NDP = self.NDP
        config = self.config

        NDP.gradient_setup(optimized_level=-1)

        mesh_vert = torch.from_numpy(np.asarray(points, dtype=np.float32)).to(config.device)
        mesh_vert = mesh_vert - self.src_mean
        warped_vert, data = NDP.warp(mesh_vert)
        warped_vert = warped_vert + self.tgt_mean
        warped_vert = warped_vert.detach().cpu().numpy()

        return warped_vert


def sample_mesh(src_mesh,samples, xrange= None,  zrange= None, vis=False):
    '''
    Args:
        S: path to mesh
    Returns:
    '''

    pcd1 =  src_mesh.sample_points_uniformly(number_of_points=samples)
    src_pcd = np.asarray(pcd1.points, dtype=np.float32)


    if xrange is not None:
        a,b = xrange

        mask =  np.logical_and( src_pcd[:,0] >= a,  src_pcd[:,0] <= b)
        src_pcd = src_pcd[mask]

    if zrange is not None:
        a,b = zrange

        mask =  np.logical_and( src_pcd[:,2] >= a,  src_pcd[:,2] <= b)
        src_pcd = src_pcd[mask]


    if vis:

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(src_pcd)
        pc.paint_uniform_color([1, 0, 0])

        src_mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([pc, src_mesh])

    return src_pcd

def process(S,T,dumpdir,type = 4):
    setup_seed(0)

    config = {
        "gpu_mode": True,
        "iters": 500,
        "lr": 0.005,
        "max_break_count": 25,
        "break_threshold_ratio": 0.0003,
        "samples": 24000,
        "motion_type": "Sim3",
        "rotation_format": "euler",
        "m": 8,
        "k0": -5,
        "depth": 3,
        "width": 128,
        "act_fn": "relu",
        "w_reg": 0,
        "w_ldmk": 10,
        "w_cd": 1
    }

    config = edict(config)
    deformer = Deformer2(config)
    src_mesh = o3d.io.read_triangle_mesh( S )
    src_mesh.paint_uniform_color([1, 0.5, 0])
    src_mesh.compute_vertex_normals()

    tgt_mesh = o3d.io.read_triangle_mesh( T )
    tgt_mesh.paint_uniform_color([0.5, 1, 0])
    tgt_mesh.compute_vertex_normals()


    trns = np.eye(4)
    if type == 1:
        trns[0][3] = -1.5
        src_mesh.transform( trns)
    if type == 2:
        trns[2][3] = -1.5
        src_mesh.transform( trns)
    if type == 3:
        trns[0][3] = 1.5
        src_mesh.transform( trns)
    if type == 4:
        trns[2][3] = 1.5
        src_mesh.transform( trns)
    if type == 1:
        src_pcd = sample_mesh(src_mesh, config.samples, xrange=[-1, -0.5])
        tgt_pcd = sample_mesh(tgt_mesh, config.samples, xrange=[-1, -0.5])
        landmarks = sample_mesh(src_mesh, config.samples // 2, xrange=[-2.5, -1])
    if type == 2:
        src_pcd = sample_mesh(src_mesh, config.samples, zrange=[-1, -0.5])
        tgt_pcd = sample_mesh(tgt_mesh, config.samples, zrange=[-1, -0.5])
        landmarks = sample_mesh(src_mesh, config.samples // 2, zrange=[-2.5, -1])
    if type == 3:
        src_pcd = sample_mesh(src_mesh, config.samples, xrange=[0.5, 1])
        tgt_pcd = sample_mesh(tgt_mesh, config.samples, xrange=[0.5, 1])
        landmarks = sample_mesh(src_mesh, config.samples // 2, xrange=[1, 2.5])
    if type == 4:
        src_pcd = sample_mesh(src_mesh, config.samples, zrange=[0.5, 1])
        tgt_pcd = sample_mesh(tgt_mesh, config.samples, zrange=[0.5, 1])
        landmarks = sample_mesh(src_mesh, config.samples // 2, zrange=[1, 2.5])

    if not np.size(src_pcd) or not np.size(tgt_pcd):
        print("skip")
        name = S.split('/')[-1]

        if type == 1:
            trns[0][3] = 1.5
            src_mesh.transform(trns)
        if type == 2:
            trns[2][3] = 1.5
            src_mesh.transform(trns)
        if type == 3:
            trns[0][3] = -1.5
            src_mesh.transform(trns)
        if type == 4:
            trns[2][3] = -1.5
            src_mesh.transform(trns)
        o3d.io.write_triangle_mesh(os.path.join(dumpdir, name + '.ply'), src_mesh)
        return

    if not np.size(landmarks):
        deformer.train_field(src_pcd, tgt_pcd, None)
    else:
        deformer.train_field(src_pcd, tgt_pcd, [landmarks,landmarks])


    src_mesh.paint_uniform_color([1, 0.5, 0])
    tgt_mesh.paint_uniform_color([0.5, 1, 0])

    warped_vert = deformer.warp_points(src_mesh.vertices)


    # o3d.visualization.draw_geometries([tgt_mesh, src_mesh])


    src_mesh.vertices = o3d.utility.Vector3dVector( warped_vert )
    # o3d.visualization.draw_geometries([tgt_mesh, src_mesh])

    name = S.split('/')[-1]

    if type == 1:
        trns[0][3] = 1.5
        src_mesh.transform( trns)
    if type == 2:
        trns[2][3] = 1.5
        src_mesh.transform( trns)
    if type == 3:
        trns[0][3] = -1.5
        src_mesh.transform( trns)
    if type == 4:
        trns[2][3] = -1.5
        src_mesh.transform( trns)
    o3d.io.write_triangle_mesh(os.path.join(dumpdir,name),src_mesh)


def postprocess(dir):

    layoutdir = glob.glob(os.path.join(dir, 'layout*'))[0]
    meshdir = os.path.join(dir, 'mesh')
    x, z, stride = layoutdir.split('/')[-1].split('_')[1].split('-')
    x, z, stride = int(x), int(z), int(stride)
    y = 32 #fixed
    views, ids = get_views(x, z, window_size=32, stride=stride)
    current_kown = []
    imax = len(views)
    jmax = len(views[0])
    dumpdir = os.path.join(dir, 'postprocessed')
    os.makedirs(dumpdir, exist_ok=True)
    for i in range(imax):
        for j in range(jmax):
            if os.path.exists(os.path.join(meshdir, str(i) + '_' + str(j) + '.ply')):
                current_kown.append(str(i) + '_' + str(j))
                shutil.copy(os.path.join(meshdir, str(i) + '_' + str(j) + '.ply'),os.path.join(dumpdir, str(i) + '_' + str(j) + '.ply'))
                break
        break
    while len(current_kown):
        current_activate = current_kown.pop(0)
        i,j = current_activate.split('_')
        if int(i)-1>=0 and not os.path.exists(os.path.join(dumpdir,str(int(i)-1)+'_' + str(j) + '.ply')) and os.path.exists(os.path.join(meshdir,str(int(i)-1)+'_' + j + '.ply')) :
            current_kown.append(str(int(i)-1)+'_' + j)
            print(i+'_'+j,str(int(i)-1)+'_' + j)
            process(os.path.join(meshdir, str(int(i)-1)+'_' + j + '.ply'),os.path.join(dumpdir, i + '_' + j + '.ply'), dumpdir, type=1)
        if int(j)-1>=0 and not os.path.exists(os.path.join(dumpdir,i+'_' + str(int(j)-1)+ '.ply')) and os.path.exists(os.path.join(meshdir,i+'_' + str(int(j)-1)+ '.ply')):
            current_kown.append( i+'_' + str(int(j)-1))
            print(i+'_'+j,i+'_' + str(int(j)-1))
            process(os.path.join(meshdir, i+'_' + str(int(j)-1)+ '.ply'),os.path.join(dumpdir, i + '_' + j + '.ply'), dumpdir, type=2)
        if int(i)+1<imax and not os.path.exists(os.path.join(dumpdir,str(int(i)+1)+'_' + str(j)+ '.ply')) and os.path.exists(os.path.join(meshdir,str(int(i)+1)+'_' + j+ '.ply')) :
            current_kown.append(str(int(i)+1)+'_' + j)
            print(i+'_'+j,str(int(i)+1)+'_' + j)
            process(os.path.join(meshdir, str(int(i) + 1) + '_' + j+ '.ply'),os.path.join(dumpdir, i + '_' + j+ '.ply'), dumpdir, type=3)
        if int(j)+1<jmax and not os.path.exists(os.path.join(dumpdir,i+'_' + str(int(j)+1)+ '.ply')) and os.path.exists(os.path.join(meshdir,i+'_' + str(int(j)+1)+ '.ply')):
            current_kown.append( i+'_' + str(int(j)+1))
            print(i+'_'+j,i+'_' + str(int(j)+1))
            process(os.path.join(meshdir,i+'_' + str(int(j)+1)+ '.ply'),os.path.join(dumpdir,i+'_' + j+ '.ply'),dumpdir,type=4)

    stride = 1.5
    meshes = o3d.geometry.TriangleMesh()
    for i in range(imax):
        for j in range(jmax):
            if os.path.exists(os.path.join(dumpdir, str(i) + '_' + str(j) + '.ply')):
                m = os.path.join(dumpdir, str(i) + '_' + str(j) + '.ply')
                mesh = o3d.io.read_triangle_mesh(m)
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color([1, 0.5, 0])
                mesh.translate((stride * i, 0, stride * j))
                meshes += mesh

    o3d.visualization.draw_geometries([meshes], mesh_show_back_face=True)

# if __name__ == "__main__":
#
#
#     config = {
#         "gpu_mode": True,
#         "iters": 500,
#         "lr": 0.005,
#         "max_break_count": 25,
#         "break_threshold_ratio": 0.0003,
#         "samples": 24000,
#         "motion_type": "Sim3",
#         "rotation_format": "euler",
#         "m": 8,
#         "k0": -5,
#         "depth": 3,
#         "width": 128,
#         "act_fn": "relu",
#         "w_reg": 0,
#         "w_ldmk": 10,
#         "w_cd": 1
#     }
#     config = edict(config)
#     deformer = Deformer2(config)
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-s', type=str,default='output/full/mesh/0_1.ply',help='Path to the src mesh.')
#     parser.add_argument('-t', type=str,default='output/full/mesh/0_0.ply',help='Path to the tgt mesh.')
#     parser.add_argument('-o', type=str, help='Path to output.')
#     # parser.add_argument('-m', type=int, help='pyramid level')
#     args = parser.parse_args()
#
#     # raw = args.r
#     S= args.s
#     T= args.t
#
#     print("--------------------------------------")
#     print( S )
#     print( T )
#
#
#
#
#     #sample overlap rigons
#
#     src_mesh = o3d.io.read_triangle_mesh( S )
#     src_mesh.paint_uniform_color([1, 0.5, 0])
#     src_mesh.compute_vertex_normals()
#
#     trns = np.eye(4)
#     #50
#     # trns [2][3] =  1
#     #75
#     trns[2][3] = 1.5
#     src_mesh.transform( trns)
#
#     tgt_mesh = o3d.io.read_triangle_mesh( T )
#     tgt_mesh.paint_uniform_color([0.5, 1, 0])
#     tgt_mesh.compute_vertex_normals()
#
#     o3d.visualization.draw_geometries([tgt_mesh, src_mesh])
#
#
#     # 50
#     # src_pcd = sample_mesh( src_mesh, config.samples, zrange=[ 0,1])
#     # tgt_pcd = sample_mesh( tgt_mesh, config.samples,  zrange=[ 0,1])
#
#     # 75
#     src_pcd = sample_mesh(src_mesh, config.samples, zrange=[0.5, 1])
#     tgt_pcd = sample_mesh(tgt_mesh, config.samples, zrange=[0.5, 1])
#
#     #50
#     # landmarks =  sample_mesh( src_mesh, config.samples // 2 , zrange=[ 1.5, 2])
#     #75
#     landmarks = sample_mesh(src_mesh, config.samples // 2, zrange=[1.5, 2.5])
#
#     # src_pcd = np.concatenate( [ src_pcd, persis_pcd], axis= 0)
#     # tgt_pcd = np.concatenate( [ tgt_pcd, persis_pcd] , axis= 0)
#     #
#     deformer.train_field(src_pcd, tgt_pcd, [landmarks,landmarks])
#
#
#     src_mesh.paint_uniform_color([1, 0.5, 0])
#     tgt_mesh.paint_uniform_color([0.5, 1, 0])
#
#
#
#     warped_vert = deformer.warp_points(src_mesh.vertices)
#
#
#     o3d.visualization.draw_geometries([tgt_mesh, src_mesh])
#
#
#     src_mesh.vertices = o3d.utility.Vector3dVector( warped_vert )
#     o3d.visualization.draw_geometries([tgt_mesh, src_mesh])
#
#     o3d.io.write_triangle_mesh('reconstruct.ply',src_mesh)


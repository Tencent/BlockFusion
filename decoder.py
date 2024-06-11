import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
class SdfDecoder(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=[],
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SdfDecoder, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.d_out = d_out

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)


    def forward(self, inputs):
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        if self.d_out == 3:
            x = torch.sigmoid(x)
        return x


class SdfModel(nn.Module):
    def __init__(self,config_json ):
        super().__init__()
        self.model = SdfDecoder(d_in=config_json['channels'],
                              d_out=1,
                              d_hidden=config_json['width'],
                              n_layers=config_json['n_layers'],
                              skip_in=config_json['skip_in'],
                              ).cuda()

        self.model.load_state_dict(torch.load(config_json['ckpt_path'], map_location='cuda'))
        self.model.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.specs["sdf_lr"])
        return optimizer

    def normalize_coordinate2(self, p, padding=0.1, plane='xz'):
        ''' Normalize coordinate to [0, 1] for unit cube experiments

        Args:
            p (tensor): point
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
            plane (str): plane feature type, ['xz', 'xy', 'yz']
        '''
        if plane == 'zx':
            xy = p[:, :, [2, 0]]
        elif plane == 'yx':
            xy = p[:, :, [1, 0]]
        else:
            xy = p[:, :, [1, 2]]

        return xy

    def sample_plane_feature(self, query, plane_feature, plane, padding=0.1):
        xy = self.normalize_coordinate2(query.clone(), plane=plane, padding=padding)
        xy = xy[:, :, None].float()
        # vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        # vgrid = xy - 1.0
        vgrid = xy

        sampled_feat = F.grid_sample(plane_feature, vgrid, padding_mode='border', align_corners=True,
                                     mode='bilinear').squeeze(-1)
        return sampled_feat

    def forward_with_plane_features(self, plane_features, xyz):
        '''
        plane_features: B, D*3, res, res (e.g. B, 768, 64, 64)
        xyz: B, N, 3
        '''
        point_features = self.get_points_plane_features(plane_features, xyz)  # point_features: B, N, D
        pred_sdf = self.model(point_features)
        return pred_sdf  # [B, num_points]

    def get_points_plane_features(self, plane_features, query):
        # plane features shape: batch, dim*3, 64, 64
        fea = {}
        fea['yx'], fea['zx'], fea['yz'] = plane_features[:, 0, ...], plane_features[:, 1, ...], plane_features[:, 2,
                                                                                                ...]
        # print("shapes: ", fea['xz'].shape, fea['xy'].shape, fea['yz'].shape) #([1, 256, 64, 64])
        plane_feat_sum = 0
        plane_feat_sum += self.sample_plane_feature(query, fea['yx'], 'yx')
        plane_feat_sum += self.sample_plane_feature(query, fea['zx'], 'zx')
        plane_feat_sum += self.sample_plane_feature(query, fea['yz'], 'yz')

        return plane_feat_sum.transpose(2, 1)

    def forward(self, plane_features, xyz):
        '''
        plane_features: B, D*3, res, res (e.g. B, 768, 64, 64)
        xyz: B, N, 3
        '''
        point_features = self.get_points_plane_features(plane_features, xyz)  # point_features: B, N, D
        pred_sdf = self.model(point_features)
        return pred_sdf  # [B, num_points]
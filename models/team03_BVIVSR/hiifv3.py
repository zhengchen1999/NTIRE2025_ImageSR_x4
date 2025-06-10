import torch
import torch.nn as nn
import torch.nn.functional as F

from .mymodels import *

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def compute_hi_coord(coord, n):
    coord_clip = torch.clip(coord - 1e-9, 0., 1.)
    coord_bin = ((coord_clip * 2 ** (n + 1)).floor() % 2)
    return coord_bin

@register('hiifv3')
class hiifv3(nn.Module):

    def __init__(self, encoder_spec, blocks=16, hidden_dim=256):
        super().__init__()
        self.encoder = make(encoder_spec)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)

        self.n_hi_layers = 6
        self.fc_layers = nn.ModuleList(
            [
                MLP_with_shortcut(hidden_dim * 4 + 2 + 2 if d == 0 else hidden_dim + 2,
                                  3 if d == self.n_hi_layers - 1 else hidden_dim,256) \
                for d in range(self.n_hi_layers)
            ]
        )



        self.conv0 = qkv_attn(hidden_dim, blocks)
        self.conv1 = qkv_attn(hidden_dim, blocks)

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        self.feat = self.freq(self.feat)
        return self.feat

    def query_rgb(self, coord, cell):
        feat = (self.feat)
        grid = 0

        pos_lr = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)

                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2] / 2
                rel_coord[:, 1, :, :] *= feat.shape[-1] / 2
                rel_coord_n = rel_coord.permute(0, 2, 3, 1).reshape(rel_coord.shape[0], -1, rel_coord.shape[1])

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                preds.append(feat_)
                if vx == -1 and vy == -1:
                    # Local coord
                    rel_coord_mask = (rel_coord_n > 0).float()
                    rxry = torch.tensor([rx, ry], device=coord.device)[None, None, :]
                    local_coord = rel_coord_mask * rel_coord_n + (1. - rel_coord_mask) * (rxry - rel_coord_n)

        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-2]
        rel_cell[:, 1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0];
        areas[0] = areas[3];
        areas[3] = t
        t = areas[1];
        areas[1] = areas[2];
        areas[2] = t

        for index, area in enumerate(areas):
            preds[index] = preds[index] * (area / tot_area).unsqueeze(1)

        grid = torch.cat([*preds, rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, coord.shape[1], coord.shape[2])], dim=1)

        B, C_g, H, W = grid.shape
        grid = grid.permute(0, 2, 3, 1).reshape(B, H * W, C_g)

        for n in range(self.n_hi_layers):
            hi_coord = compute_hi_coord(local_coord, n)
            if n == 0:
                x = torch.cat([grid] + [hi_coord], dim=-1)
            else:
                x = torch.cat([x] + [hi_coord], dim=-1)
            x = self.fc_layers[n](x)
            if n == 0:
                x = self.conv0(x)
                x = self.conv1(x)


        result = x.permute(0, 2, 1).reshape(B, 3, H, W)

        ret = result + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear', \
                                  padding_mode='border', align_corners=False)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MLP_with_shortcut(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        short_cut = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if x.shape[-1] == short_cut.shape[-1]:
            x = x + short_cut
        return x

class qkv_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Linear(midc, midc * 3, bias=True)

        self.kln = nn.LayerNorm(self.headc)
        self.vln = nn.LayerNorm(self.headc)
        self.sm = nn.Softmax(dim=-1)

        self.proj1 = nn.Linear(midc, midc)
        self.proj2 = nn.Linear(midc, midc)

        self.proj_drop = nn.Dropout(0.)

        self.act = nn.GELU()

    def forward(self, x):
        B, HW, C = x.shape
        bias = x

        qkv = self.qkv_proj(x).reshape(B, HW, self.heads, 3 * self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1) # B, heads, HW, headc

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2, -1), v) / (HW)
        # v = self.sm(v)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, HW, C)

        ret = v + bias
        bias = self.proj2(self.act(self.proj1(ret))) + bias

        return bias

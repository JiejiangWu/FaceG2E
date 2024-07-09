# import pytorch3d.io as torch3d_io
# import pytorch3d.loss as torch3d_loss
# import pytorch3d.structures as torch3d_struct
# from pytorch3d.ops import cot_laplacian
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
# import torch.nn as nn
from lib.boxdiff.ptp_utils import AttentionStore, aggregate_attention
import math,cv2

def attention_mask(attentionStore:AttentionStore, 
                    indices_to_alter:List[int], 
                    attention_res:int, 
                    target_res:int, 
                    soft_threshold:float):
    '''
    attentionStore: attentionStore类
    token_idx: 取第几个text token相关的attention map
    attention_res: 一般为16x16，即sd中取的attention分辨率
    target_res: 插值后用于取出激活像素的msk的分辨率
    soft_threshold: 比例，取前百分之多少的值作为阈值，大于阈值的被认为是激活的，例如 max - 0.2*(max-min)
    '''

    attention_maps = aggregate_attention(attentionStore, attention_res, from_where=('up','mid','down'), is_cross=True, select=0).detach().cpu()
    # token_attention_map = attention_maps[:,:,token_idx] # 16x16
    
    last_idx = -1
    attention_for_text = attention_maps[:, :, 1:last_idx].float()
    attention_for_text *= 100
    # with torch.cuda.amp.autocast(enabled=True):
    #     attention_for_text = torch.nn.functional.softmax(attention_for_text.float(), dim=-1)

    # Shift indices since we removed the first token
    indices_to_alter = [index - 1 for index in indices_to_alter]

    extract_msks = []
    heat_maps = []
    for i in indices_to_alter:
        attention_image = attention_for_text[:, :, i].view(1,1,attention_res,attention_res).float().detach()
        attention_image_interpolated = F.interpolate(attention_image,size=[target_res,target_res],mode='bilinear')
        
        # 计算soft阈值
        # threshold = torch.max(attention_image) - soft_threshold*(torch.max(attention_image) - torch.min(attention_image))
        # 计算msk
        # extract_msk = attention_image_interpolated > threshold

        extract_msk = (attention_image_interpolated-torch.min(attention_image_interpolated))/(torch.max(attention_image_interpolated) - torch.min(attention_image_interpolated))
        
        mask = extract_msk[0].permute(1,2,0).detach().cpu().numpy()
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = torch.tensor(np.float32(heatmap) / 255).unsqueeze(0).permute(0,3,1,2)


        extract_msks.append(extract_msk.unsqueeze(0))
        heat_maps.append(heatmap)
    return torch.cat(extract_msks,dim=0),torch.cat(heat_maps,dim=0)

def gaussian_kernel(kernel_size, sigma=2., dim=2, channels=3, device='cpu'):
  # The gaussian kernel is the product of the gaussian function of each dimension.
  # kernel_size should be an odd number.

  # kernel_size = 2 * size + 1

  kernel_size = [kernel_size] * dim
  sigma = [sigma] * dim
  kernel = torch.ones(kernel_size, device=device)
  meshgrids = torch.meshgrid(
      [torch.arange(size, dtype=torch.float32, device=device) for size in kernel_size])

  for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
    mean = (size - 1) / 2
    kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) /
                                                               (2 * std))**2)

  # Make sure sum of values in gaussian kernel equals 1.
  kernel = kernel / torch.sum(kernel)

  # Reshape to depthwise convolutional weight
  kernel = kernel.view(1, 1, *kernel.size())
  kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

  return kernel


def gaussian_blur(x, kernel_size, sigma=None, dim=None, channels=None):
  if sigma is None:
    sigma = kernel_size / 3
  if dim is None:
    dim = len(x.shape) - 2
  if channels is None:
    channels = x.shape[1]
  kernel = gaussian_kernel(kernel_size, sigma, dim, channels, x.device)
  # kernel_size = 2 * size + 1

  # x = x[None, ...]
  padding = int((kernel_size - 1) / 2)
  x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
  # x = torch.squeeze(F.conv2d(x, kernel, groups=channels))
  x = F.conv2d(x, kernel, groups=channels)

def texture_symmetric_loss(uv_map,use_blur=False):
    '''
    uv_map B,C,H,W
    '''
    uv_size = uv_map.shape[2]
    if use_blur:
        process_uv_map = gaussian_blur(uv_map, uv_size // 64 + 1)
    else:
        process_uv_map = uv_map
    flipped_uv = torch.flip(process_uv_map, dims=(3,)).detach()
    return F.smooth_l1_loss(process_uv_map, flipped_uv)

def texture_smooth_loss(uv_map):
    '''
    uv_map B,C,H,W
    '''
    nch = uv_map.shape[1]
    w = torch.FloatTensor([
    [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, -1, 0]]
    ]).unsqueeze(1).to(uv_map.device)

    diff = []    
    for ch in range(nch):
        ch_diff = torch.nn.functional.conv2d(uv_map[:,ch:ch+1],w) 
        ch_diff = ch_diff**2
        diff.append(ch_diff)
    total_diff = torch.stack(diff,dim=1).sum(dim=1)
    return torch.abs(total_diff).mean() / nch


# def laplacian_smoothing_loss(verts,triangles,method='uniform'):
#     meshes = torch3d_struct.Meshes(verts=[verts],faces=[triangles])

#     # https://github.com/facebookresearch/pytorch3d/blob/995b60e3b99faa1ee1bcdbe244426d54d98a7242/pytorch3d/loss/mesh_laplacian_smoothing.py
#     if meshes.isempty():
#         return torch.tensor(
#             [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
#         )

#     N = len(meshes)
#     verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
#     faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
#     num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
#     verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
#     weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
#     weights = 1.0 / weights.float()

#     # We don't want to backprop through the computation of the Laplacian;
#     # just treat it as a magic constant matrix that is used to transform
#     # verts into normals
#     with torch.no_grad():
#         if method == "uniform":
#             L = meshes.laplacian_packed()
#         elif method in ["cot", "cotcurv"]:
#             L, inv_areas = cot_laplacian(verts_packed, faces_packed)
#             if method == "cot":
#                 norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
#                 idx = norm_w > 0
#                 # pyre-fixme[58]: `/` is not supported for operand types `float` and
#                 #  `Tensor`.
#                 norm_w[idx] = 1.0 / norm_w[idx]
#             else:
#                 L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
#                 norm_w = 0.25 * inv_areas
#         else:
#             raise ValueError("Method should be one of {uniform, cot, cotcurv}")

#     if method == "uniform":
#         loss = L.mm(verts_packed)
#     elif method == "cot":
#         loss = L.mm(verts_packed) * norm_w - verts_packed
#     elif method == "cotcurv":
#         # pyre-fixme[61]: `norm_w` may not be initialized here.
#         loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w
#     loss = loss.norm(dim=1)

#     loss = loss * weights
#     # https://github.com/facebookresearch/pytorch3d/issues/432
#     # return loss.sum() / N
#     return torch.linalg.norm(loss) / N



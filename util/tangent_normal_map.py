from email.policy import default
import torch
import torch.nn.functional as F
import kornia
from kornia.geometry.camera import pixel2cam
import numpy as np
from typing import List
# import nvdiffrast.torch as dr
from scipy.io import loadmat
import scipy,os,sys
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from models import hifi3dmm


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse).long()
    return torch.gather(input, dim, index)

def compute_TBN(vertex_world_coordinates, vertex_world_normals, texl_uv_coordinates, tri, tri_vts, point_buf):
    '''
    parameter:  vertex_world_coordinates:   B * Nv * 3
                texl_uv_coordinates:        B * Nt * 3
                triangles:                  B * Nf * 3
                triangles:                  B * Nf * 3
                point_buf:                  B * Nv * 8 (A single vertex can be shared by up to 8 triangles in HIFI3D++ topology)

    return:     vertex_TBN:                 B * Nf * 3*3
    '''
    batch_num = vertex_world_coordinates.shape[0]
    vertex_num = vertex_world_coordinates.shape[1]
    # propress the input, make sure BATCH-style
    if texl_uv_coordinates.dim() == 2:
        texl_uv_coordinates = texl_uv_coordinates.unsqueeze(0).repeat(batch_num,1,1)
    assert texl_uv_coordinates.dim() == 3
    if tri.dim() == 2:
        tri = tri.unsqueeze(0).repeat(batch_num,1,1)
    assert tri.dim() == 3
    if tri_vts.dim() == 2:
        tri_vts = tri_vts.unsqueeze(0).repeat(batch_num,1,1)
    assert tri_vts.dim() == 3
    texl_num = texl_uv_coordinates.shape[1]


    # https://zhuanlan.zhihu.com/p/139593847
    # e1 = vertex_world_coordinates[tri[...,1]] - vertex_world_coordinates[tri[...,0]]
    # e2 = vertex_world_coordinates[tri[...,2]] - vertex_world_coordinates[tri[...,0]]
    # delta1 = texl_uv_coordinates[tri_vts[...,1]] - texl_uv_coordinates[tri_vts[...,0]]
    # delta2 = texl_uv_coordinates[tri_vts[...,2]] - texl_uv_coordinates[tri_vts[...,0]]
    e1 = batched_index_select(input=vertex_world_coordinates,dim=1,index=tri[...,1]) - batched_index_select(input=vertex_world_coordinates,dim=1,index=tri[...,0])
    e2 = batched_index_select(input=vertex_world_coordinates,dim=1,index=tri[...,2]) - batched_index_select(input=vertex_world_coordinates,dim=1,index=tri[...,0])
    delta1 = batched_index_select(input=texl_uv_coordinates,dim=1,index=tri_vts[...,1]) - batched_index_select(input=texl_uv_coordinates,dim=1,index=tri_vts[...,0])
    delta2 = batched_index_select(input=texl_uv_coordinates,dim=1,index=tri_vts[...,2]) - batched_index_select(input=texl_uv_coordinates,dim=1,index=tri_vts[...,0])
    # e1,e2:            B * Nf * 3
    # delta1,delta2:    B * Nf * 2


    delta_u1 = delta1[...,0].unsqueeze(-1)    # B * Nf * 1
    delta_v1 = delta1[...,1].unsqueeze(-1)    # B * Nf * 1
    delta_u2 = delta2[...,0].unsqueeze(-1)    # B * Nf * 1
    delta_v2 = delta2[...,1].unsqueeze(-1)    # B * Nf * 1
    # per-triangle tangent
    face_tangent = (delta_v1*e2 - delta_v2*e1) / (delta_v1*delta_u2 - delta_v2*delta_u1)    # B * Nf * 1
    # per-vertex tangent
    face_tangent_norm = F.normalize(face_tangent,dim=-1)
    face_tangent_norm = torch.cat([face_tangent_norm,torch.zeros(face_tangent_norm.shape[0],1,3).to(face_tangent_norm.device)], dim=1)  # B * (Nf+1) * 1

    vertex_tangent = torch.sum(face_tangent_norm[:, point_buf], dim=2) # B * Nv * 1
    vertex_tangent_norm = F.normalize(vertex_tangent, dim=-1, p=2)             # B * Nv * 1
    
    # orthogonalization
    # tangent = F.normalize(vertex_tangent_norm - torch.tensordot(vertex_tangent_norm,vertex_world_normals,dims=2) * vertex_world_normals)
    tangent = F.normalize(vertex_tangent_norm - torch.einsum('bijk,bikl->bijl',[vertex_tangent_norm.unsqueeze(-2),vertex_world_normals.unsqueeze(-1)]).squeeze(-1) * vertex_world_normals, dim=2)

    # bitangent
    bitangent = torch.cross(tangent,vertex_world_normals)

    #TODO return vertex_TBN
    return tangent, bitangent

def world_to_Tangent(T,B,N):
    '''
    parameter:  T,B,N: the tbn_coord_basis,     B * Nv * 3

    return:     world2tbnMatrix:                B * Nv * 3*3
    '''
    
    return torch.inverse(Tangent_to_world(T,B,N))
def Tangent_to_world(T,B,N):
    '''
    parameter:  T,B,N: TBN_coord_basis:     B * Nv * 3

    return:     tbn2worldMatrix:            B * Nv * 3*3
    '''
    return torch.cat([T.unsqueeze(-1),B.unsqueeze(-1),N.unsqueeze(-1)],dim=-1)



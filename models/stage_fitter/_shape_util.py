from cgitb import grey
from gettext import translation
import re
import numpy as np
from models.hifi3dmm import HIFIParametricFaceModel
import torch.nn as nn
import torch
from util.nvdiffrast import MeshRenderer
import torch.nn.functional as F
import random
from util.render_util import random_gamma
from util import meshio
import nvdiffrast.torch as dr

def reverse_rasterize(uv_coord_input,uv_idx_input,tri_idx,rasterize_size=512):
    '''
    uv_coord_input: uv coordinates in the mesh          uvx2
    uv_idx_input:   uv index of each mesh surface    fx3
    tri_idx     :   index of the vertices of each mesh  fx3
    '''
    device ='cuda'
    vertex_ndc = torch.cat([uv_coord_input.unsqueeze(0)*2-1,0.5*torch.ones(1,uv_coord_input.shape[0],1).to(device),torch.ones(1,uv_coord_input.shape[0],1).to(device)], dim=2).contiguous()
    tri = uv_idx_input.contiguous()
    
    vertex_ndc = torch.cat([uv_coord_input[:,0:1].unsqueeze(0)*2-1,
                            (1-uv_coord_input[:,1:2]).unsqueeze(0)*2-1,
                            0.5*torch.ones(1,uv_coord_input.shape[0],1).to(device),
                            torch.ones(1,uv_coord_input.shape[0],1).to(device)], dim=2).contiguous()

    try: 
        glctx = dr.RasterizeCudaContext(device=device)
        print("create glctx on device cuda")
    except:
        glctx = dr.RasterizeGLContext(device=device)
        print("create glctx on device cuda")
    
    ranges = None

    tri = tri.type(torch.int32).contiguous()
    vertex_ndc = vertex_ndc.type(torch.float32).contiguous()
    # why max rast_out[0,:,:,3] = 13000?
    rast_out, _ = dr.rasterize(glctx, vertex_ndc.contiguous(), tri, resolution=[rasterize_size, rasterize_size], ranges=ranges)

    return rast_out

def predict_dp(self):
    self.dp_map = F.tanh(self.dp_tensor) * self.dp_map_scale


def update_shape(self):
    output_coeff = self.concat_coeff()
    self.predict_dp()
    if self.stage != 'coarse geometry generation':
        self.pred_vertex, self.pred_vertex_norm = \
            self.facemodel.compute_for_render_with_dp_map(output_coeff,self.dp_map,use_external_exp=False)    # torch.Size([1, 20481, 3])
    else:
        self.pred_vertex, self.texture_3dmm, self.pred_vertex_norm = \
            self.facemodel.compute_for_render(output_coeff,use_external_exp=False)    # torch.Size([1, 20481, 3])
    self.pred_vertex_no_pose = self.pred_vertex.clone()

def shape2posmap(self,vert):
    
    # pre-process
    vert_world = self.facemodel.to_world(vert)  # -1.5 ~ 1.5


    if not hasattr(self,'reverse_rast_out'):
        self.reverse_rast_out = reverse_rasterize(self.facemodel.vt,self.facemodel.complete_face_vt,self.facemodel.tri,rasterize_size=512)
    rast_out = self.reverse_rast_out
    bary_x = rast_out[0,:,:,0:1]
    bary_y = rast_out[0,:,:,1:2]
    bary_z = 1 - bary_x - bary_y

    bray_new = torch.cat([bary_x,bary_y,bary_z],dim=-1) # Calculate trigonometric coordinates for each uv pixel
    face_idx = rast_out[0,:,:,3]
    tri = torch.tensor(self.facemodel.tri).to('cuda')
    pix_wise_tri_pos = vert_world[tri[face_idx.int()-1]]              # 512x512x3x3 # Each uv pixel corresponds to the three vertex positions of the triangle
    pix_wise_pos_map = torch.sum(bray_new.unsqueeze(3) * pix_wise_tri_pos,dim=-2) # Spatial pos position corresponding to each uv pixel
    pix_wise_pos_msk = (face_idx>0)
    pix_wise_pos_map *= pix_wise_pos_msk.unsqueeze(-1)
    return pix_wise_pos_map
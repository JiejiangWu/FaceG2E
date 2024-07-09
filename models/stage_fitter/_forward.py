from cgitb import grey
from gettext import translation
import numpy as np
from models.hifi3dmm import HIFIParametricFaceModel
import torch.nn as nn
import torch
from util.nvdiffrast import MeshRenderer
import torch.nn.functional as F
import random
from util.render_util import random_gamma
from util import meshio
from util import util

def generate_diffuse(self):
    if self.diffuse_generation_type == 'latent':
        with torch.cuda.amp.autocast(enabled=True):
            self.diffuse_texture = util.SD_decode_latents(self.guidance,self.diffuse_latent).float().permute(0,2,3,1).contiguous()
    elif self.diffuse_generation_type == 'direct':
        self.diffuse_texture = self.diffuse_texture

def encode_diffuse_latent(self):
    with torch.cuda.amp.autocast(enabled=True):
        self.diffuse_latent[:] = util.SD_encode_imgs(self.guidance,self.diffuse_texture.permute(0,3,1,2)).float().contiguous()[:]


def compute_facial_mask(self):
    pred_mask, _, _, _ = self.renderer.forward_with_texture_map(
            self.pred_vertex, self.facemodel.face_buf[self.facemodel.face_region_tri_idx], self.diffuse_texture, self.facemodel.vt, self.facemodel.face_vt[self.facemodel.face_region_tri_idx], self.pred_vertex_norm)
    return pred_mask

def render_latent(self,latent_res=64):
    origin_render_res = self.renderer.rasterize_size
    # Modify the render resolution setting
    self.renderer.rasterize_size = latent_res
    pix_msk, pix_depth, pix_latent, pix_normal = self.renderer.forward_with_texture_map(
                self.pred_vertex, self.facemodel.face_buf, self.diffuse_latent.permute(0,2,3,1).contiguous(), self.facemodel.vt, self.facemodel.face_vt, self.pred_vertex_norm)
    # Reset the render resolution setting
    self.renderer.rasterize_size = origin_render_res
    return pix_msk, pix_depth, pix_latent, pix_normal

def render_control_img(self,control_res=512):
    origin_render_res = self.renderer.rasterize_size
    # Modify the render resolution setting
    self.renderer.rasterize_size = control_res
    pix_msk, pix_depth, _, pix_normal = self.renderer.forward_with_texture_map(
                self.pred_vertex, self.facemodel.face_buf, self.constant_diffuse_texture, self.facemodel.vt, self.facemodel.face_vt, self.pred_vertex_norm)
    # Reset the render resolution setting
    self.renderer.rasterize_size = origin_render_res
    return pix_msk, pix_depth, None, pix_normal    

def forward(self,random_sample_view=True,render_latent=False,random_sample_light=True,render_origin_diffuse=False):
    '''
    random_sample: Whether random lighting + random pose is used for rendering
    render_latent: Whether to use the latent rendering, that is, the latent (64x64x4) effect of the diffuse directly rendering (64x64x4)
    '''
    
    output_coeff = self.concat_coeff()

    self.update_shape()
    # TODO animation stage是不是没有用
    # generate diffuse
    if self.stage != 'animation':
        self.generate_diffuse()

    # randomly sample new view point
    if random_sample_view:
        self.rotation,self.translation_z = self.random_transformation(with_choice= self.random_view_with_choice)

        if self.stage == 'animation':
            self.rotation = [0,0,0]
            self.translation_z = 3

    # transform viewpoint
    self.apply_transformation(self.rotation,self.translation_z)
    
    # random light
    if random_sample_light:
        self.random_light()

    # Render the original diffuse+geometry result if needed
    if render_origin_diffuse:
        # calculate origin geometry(origin_pred_vertex)
        origin_pred_vertex = self.compute_transformed_vertex(self.load_vertex, self.rotation,self.translation_z)

        # calcuate diffuse rendered
        _, _, origin_pix_tex, pix_norm = self.renderer.forward_with_texture_map(
                origin_pred_vertex, self.facemodel.face_buf, self.origin_diffuse_texture, self.facemodel.vt, self.facemodel.face_vt, self.origin_pred_vertex_norm)
        self.pred_face_origin_diffuse = self.facemodel.compute_color(origin_pix_tex.permute(0,2,3,1),pix_norm.permute(0,2,3,1),output_coeff,white_light=False) #pred_face:[bs,imgsize,imgsize,3]
        self.pred_face_origin_diffuse = self.pred_face_origin_diffuse.contiguous().permute(0,3,1,2)

        # calcuate textureless rendered
        _, _, origin_grey_pix_tex, _ = self.renderer.forward_with_texture_map(
                origin_pred_vertex, self.facemodel.face_buf, self.constant_diffuse_texture, self.facemodel.vt, self.facemodel.face_vt, self.origin_pred_vertex_norm)        
        self.pred_face_origin_grey = self.facemodel.compute_color(origin_grey_pix_tex.permute(0,2,3,1),pix_norm.permute(0,2,3,1),output_coeff,white_light=False) #pred_face:[bs,imgsize,imgsize,3]
        self.pred_face_origin_grey = self.pred_face_origin_grey.contiguous().permute(0,3,1,2)

        # calculate normal rendered
        pix_norm = F.normalize(pix_norm,dim=1)
        self.pred_face_origin_normal = pix_norm * 0.5 + 0.5
    else:
        self.pred_face_origin_diffuse = None
        self.pred_face_origin_grey = None
        self.pred_face_origin_normal = None

    # 如果是latent render模式
    if render_latent:
        _, _, rendered_latent, _ = self.render_latent() # 64x64, latent mode

        return rendered_latent, None,None,None,None


    # Calculate the rasterization result of the full gray map
    _, _, grey_pix_tex, pix_norm = self.renderer.forward_with_texture_map(
                self.pred_vertex, self.facemodel.face_buf, self.constant_diffuse_texture, self.facemodel.vt, self.facemodel.face_vt, self.pred_vertex_norm)

    pix_norm = F.normalize(pix_norm,dim=1)


    # Calculate the rasterization result of the duffuse texture map
    if self.diffuse_generation_type == 'mlp':
        self.pred_mask, self.pred_depth, pix_tex, pix_norm = self.renderer.forward_with_texture_mlp(
                self.pred_vertex_no_pose,            
                self.pred_vertex, self.facemodel.face_buf, self.diffuse_mlp, self.pred_vertex_norm)
    else:
        self.pred_mask, self.pred_depth, pix_tex, pix_norm = self.renderer.forward_with_texture_map(
                self.pred_vertex, self.facemodel.face_buf, self.diffuse_texture, self.facemodel.vt, self.facemodel.face_vt, self.pred_vertex_norm)

    pix_norm = F.normalize(pix_norm,dim=1)
    self.grey_pix_tex = grey_pix_tex
    self.pix_tex = pix_tex
    self.pix_norm = pix_norm
    # Render full gray mode results
    self.grey_pred_face = self.facemodel.compute_color(grey_pix_tex.permute(0,2,3,1),pix_norm.permute(0,2,3,1),output_coeff,white_light=False) #pred_face:[bs,imgsize,imgsize,3]
    self.grey_pred_face = self.grey_pred_face.contiguous().permute(0,3,1,2)

    # Render diffuse texture map results
    self.pred_face = self.facemodel.compute_color(pix_tex.permute(0,2,3,1),pix_norm.permute(0,2,3,1),output_coeff,white_light=False) #pred_face:[bs,imgsize,imgsize,3]
    self.pred_face = self.pred_face.contiguous().permute(0,3,1,2)


    return self.pred_face, self.grey_pred_face, self.pred_depth, self.pix_norm, self.pred_mask
    

def render_specific_view(self,rotation=[0,0,0],translation_z=0,input_gamma=None):
    with torch.no_grad():
        # set light
        if input_gamma!=None:
            self.set_gamma(input_gamma)
        else:
            self.set_gamma(torch.tensor([0.6,-0.2,0.25,-0.15,0,-0.15,0,0,0] * 3)[None, ...].float().to(self.device))

        output_coeff = self.concat_coeff()

        self.pred_vertex = self.pred_vertex_no_pose.detach().clone()
        
        self.pred_vertex_norm = self.facemodel.compute_norm(self.facemodel.to_world(self.pred_vertex_no_pose.clone()))  # Rotation in camera coordinates does not affect the normal direction in world coordinates

        self.apply_transformation(rotation,translation_z)      

        # grey tex
        pred_mask, pred_depth, pix_grey_tex, pix_norm = self.renderer.forward_with_texture_map(
                self.pred_vertex, self.facemodel.face_buf, self.constant_diffuse_texture, self.facemodel.vt, self.facemodel.face_vt, self.pred_vertex_norm)            
        # diffuse tex
        if self.diffuse_generation_type == 'mlp':
            pred_mask, pred_depth, pix_tex, pix_norm = self.renderer.forward_with_texture_mlp(
                    self.pred_vertex_no_pose,            
                    self.pred_vertex, self.facemodel.face_buf, self.diffuse_mlp, self.pred_vertex_norm)
        else:
            pred_mask, pred_depth, pix_tex, pix_norm = self.renderer.forward_with_texture_map(
                    self.pred_vertex, self.facemodel.face_buf, self.diffuse_texture, self.facemodel.vt, self.facemodel.face_vt, self.pred_vertex_norm)

        pix_norm = F.normalize(pix_norm,dim=1)

        # grey face
        pred_grey_face = self.facemodel.compute_color(pix_grey_tex.permute(0,2,3,1),pix_norm.permute(0,2,3,1),output_coeff,white_light=False) #pred_face:[bs,imgsize,imgsize,3]
        pred_grey_face = pred_grey_face.contiguous().permute(0,3,1,2)
        pred_face = self.facemodel.compute_color(pix_tex.permute(0,2,3,1),pix_norm.permute(0,2,3,1),output_coeff,white_light=False) #pred_face:[bs,imgsize,imgsize,3]
        pred_face = pred_face.contiguous().permute(0,3,1,2)

        return pred_face, pred_grey_face, pred_depth, pix_norm, pix_tex, pred_mask

def bi_direction_projection(self,rotation,translation_z,input_map,direction='obverse',interpolate_mode='linear',specific_img_size=0, specific_uv_size=0):
    self.update_shape()
    self.apply_transformation(rotation,translation_z)
    
    ndc_proj = self.renderer.ndc_proj.to('cuda')
    if self.pred_vertex.shape[-1] == 3:
        vertex = torch.cat([self.pred_vertex, torch.ones([*self.pred_vertex.shape[:2], 1]).to('cuda')], dim=-1)
        vertex[..., 1] = -vertex[..., 1] 
    vertex_ndc = vertex @ ndc_proj.t()
    
    pred_mask,pix_tex = self.renderer.bi_direction_forward(vertex_ndc, self.facemodel.face_buf, input_map, self.facemodel.vt, self.facemodel.face_vt,
                            interpolate_mode=interpolate_mode,direction=direction,
                            specific_img_size=specific_img_size, specific_uv_size=specific_uv_size)

    return pred_mask,pix_tex
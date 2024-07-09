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
import os
from util.io_util import save_tensor2img
from util.util import mean_list,write_obj,list2str
from PIL import Image
import lib.boxdiff.vis_utils as vis_utils

def load_shape(self,coeff_path,dp_path=None):
    with torch.no_grad():
        if coeff_path:
            self.id_para[:] = torch.tensor(np.load(coeff_path)).float().to(self.device)[:]
    
        if dp_path != None:
            self.dp_tensor[:] = torch.tensor(np.load(dp_path)).float().to(self.device)[:]

        # 保存载入的id para和顶点位置
        self.load_id_para = self.id_para.clone()
        self.update_shape()
        self.load_vertex = self.pred_vertex_no_pose.clone()
        # 保存载入的id para对应的vertex_norm
        self.origin_pred_vertex_norm = self.pred_vertex_norm.clone()

def load_diffuse(self,diffuse_path=None):
    if diffuse_path:
        if self.diffuse_generation_type == 'direct':
            self.diffuse_texture[:] = torch.tensor(np.array(Image.open(diffuse_path,'r').convert('RGB')) / 255.).unsqueeze(0).float().contiguous().to(self.device)[:]
        if self.diffuse_generation_type == 'latent':
            if diffuse_path[-4:] == '.npy':
                self.diffuse_latent[:] = torch.tensor(np.load(diffuse_path)).float().contiguous().to(self.device)[:]
                self.generate_diffuse()
            if diffuse_path[-4:] in ['.png','.jpg','.jpeg']:
                self.diffuse_texture = torch.tensor(np.array(Image.open(diffuse_path,'r').convert('RGB').resize([512,512])) / 255.).unsqueeze(0).float().contiguous().to(self.device)[:]
                self.encode_diffuse_latent()
    self.origin_diffuse_texture = self.diffuse_texture.clone()



def render_uv_from_mlp(self):
    if self.diffuse_generation_type != 'mlp':
        return

    from lib.fantasia3d.render import texture,mlptexture,mesh,material,render
    import nvdiffrast.torch as dr
    from PIL import Image
    from lib.fantasia3d.geometry.dlmesh import DLMesh
    glctx = dr.RasterizeCudaContext()

#  v_pos=None, t_pos_idx=None, v_nrm=None, t_nrm_idx=None, v_tex=None, t_tex_idx=None, v_tng=None, t_tng_idx=None, material=None, base=Non

    # base_mesh = mesh(v_pos=self.pred_vertex_no_pose[0],v_tex=self.face)
    # geometry = DLMesh(base_mesh, None)
    new_mesh = mesh.Mesh(v_tex=self.facemodel.vt.contiguous(), t_tex_idx=self.facemodel.face_vt.contiguous(), 
                            v_pos=self.pred_vertex_no_pose[0].contiguous(), t_pos_idx=self.facemodel.tri.contiguous())
    kd=render.render_diffuse_uv(glctx, new_mesh, [1024,1024], self.diffuse_mlp)
    self.diffuse_texture = kd.clone()


def save_visuals(self,exp_folder,iter_step,rx=0,ry=0,rz=0,tz=0):
    # 当前视角下的渲染
    # self.pred_face, self.grey_pred_face, self.pred_depth, self.pix_norm
    if self.diffuse_generation_type=='mlp':
        self.render_uv_from_mlp()
    save_tensor2img(os.path.join(exp_folder,f'{iter_step}_rendered.png'), self.pred_face)
    save_tensor2img(os.path.join(exp_folder,f'{iter_step}_grey_rendered.png'), self.grey_pred_face)
    save_tensor2img(os.path.join(exp_folder,f'{iter_step}_norm.png'), self.pix_norm * 0.5 + 0.5)
    save_tensor2img(os.path.join(exp_folder,f'{iter_step}_diffuse.png'), self.diffuse_texture)
    save_tensor2img(os.path.join(exp_folder,f'{iter_step}_diffuse_tensor.png'), self.diffuse_latent[:,:3])

    if hasattr(self,'control_img') and self.control_img != None:
        save_tensor2img(os.path.join(exp_folder,f'{iter_step}_control.png'), self.control_img)

    if hasattr(self,'UV_attention_weight'): save_tensor2img(os.path.join(exp_folder,f'{iter_step}_UV_att_weight.png'), self.UV_attention_weight)
    if hasattr(self,'tmp_UV_weight'): save_tensor2img(os.path.join(exp_folder,f'{iter_step}_tmp_UV_weight.png'), self.tmp_UV_weight)
    if hasattr(self,'update_pos_map'): save_tensor2img(os.path.join(exp_folder,f'{iter_step}_update_pos_map.png'), self.update_pos_map *0.4 + 1) 
    if hasattr(self,'origin_pos_map'): save_tensor2img(os.path.join(exp_folder,f'{iter_step}_origin_pos_map.png'), self.origin_pos_map *0.4 + 1) 


    # 固定视角下的渲染
    pred_face, pred_grey_face, pred_depth, pix_norm, pix_tex, pred_mask = self.render_specific_view([rx,ry,rz]
    # pred_face, pred_grey_face, pix_norm = self.render_specific_view([rx,ry,rz]
                                                                    ,tz)

    save_tensor2img(os.path.join(exp_folder,'display',f'{iter_step}_rendered.png'), pred_face)
    save_tensor2img(os.path.join(exp_folder,'display',f'{iter_step}_grey_rendered.png'), pred_grey_face)
    save_tensor2img(os.path.join(exp_folder,'display',f'{iter_step}_norm.png'), pix_norm * 0.5 + 0.5)

def save_attention(self,exp_folder,iter_step,text):
    if self.guidance.attentionStore != None:
        n = len(self.guidance.tokenizer.encode(text))
        orig_image = Image.fromarray((self.pred_face[0].permute(1,2,0).detach().cpu().numpy() * 255.).clip(0,255).astype(np.uint8))
        vis_img = vis_utils.show_cross_attention(text, self.guidance.attentionStore, self.guidance.tokenizer, np.arange(n), res=16, from_where=("up", "down", "mid"), orig_image=orig_image)

        os.makedirs(exp_folder,exist_ok=True)
        vis_img.save(os.path.join(exp_folder,f'{iter_step}_attention.png'))


def save_results(self,exp_folder,iter_step,save_mesh=True,save_npy=True):
    if save_mesh:
        recon_shape = self.pred_vertex[0].clone()  # get reconstructed shape
        recon_shape[..., -1] = self.camera_d - self.pred_vertex[0][..., -1] # from camera space to world space
        write_obj(os.path.join(exp_folder,f'{iter_step}.obj'),
                    recon_shape.detach().cpu().numpy(),
                    self.facemodel.vt.detach().cpu().numpy(),
                    self.facemodel.tri.detach().cpu().numpy(),
                    self.facemodel.complete_face_vt.detach().cpu().numpy(),
                    os.path.join(exp_folder,f'{iter_step}_diffuse.png')
                    )

        recon_shape = self.pred_vertex_no_pose[0].clone()  # get reconstructed shape
        recon_shape[..., -1] = self.camera_d - self.pred_vertex_no_pose[0][..., -1] # from camera space to world space
        write_obj(os.path.join(exp_folder,f'{iter_step}_no_pose.obj'),
                    recon_shape.detach().cpu().numpy(),
                    self.facemodel.vt.detach().cpu().numpy(),
                    self.facemodel.tri.detach().cpu().numpy(),
                    self.facemodel.complete_face_vt.detach().cpu().numpy(),
                    os.path.join(exp_folder,f'{iter_step}_diffuse.png')
                    )
    if save_npy:
        # coeff保存
        np.save(os.path.join(exp_folder,f'{iter_step}_coeff.npy'),self.id_para.detach().cpu().numpy())
        # diffuse tensor保存
        if self.diffuse_generation_type == 'latent':
            np.save(os.path.join(exp_folder,f'{iter_step}_diffuse_latent.npy'), self.diffuse_latent.detach().cpu().numpy())
        # dp tensor保存
        np.save(os.path.join(exp_folder,f'{iter_step}_dp.npy'),self.dp_tensor.detach().cpu().numpy())
        save_tensor2img(os.path.join(exp_folder,f'{iter_step}_dp.png'), self.dp_map/self.dp_map_scale)

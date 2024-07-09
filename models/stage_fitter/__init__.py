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

def degree2radian(angle):
    return 0.0174532925 * angle

class StageFitter(object):
    def __init__(self, SD_guidance,
                stage='coarse geometry generation',
                diffuse_generation_type = 'direct',
                render_resolution=224,fov=12.593637,camera_d=10,
                texture_resolution=512, dp_map_resolution=128,
                device='cuda',
                saved_id_path = None,
                saved_dp_path = None,
                saved_diffuse_path = None,
                latent_init='zeros',
                dp_map_scale=0.0025,
                edit_scope='tex',
                ):
        self.stage = stage
        self.guidance = SD_guidance
        self.diffuse_generation_type = diffuse_generation_type
        # camera setting
        self.resolution = render_resolution
        self.fov = fov
        self.camera_d = camera_d
        center = self.resolution / 2
        self.focal = center / np.tan(self.fov * np.pi / 360)
        self.edit_scope = edit_scope
        # hifi 3dmm
        self.facemodel = HIFIParametricFaceModel(
                hifi_folder='./HIFI3D', camera_distance=self.camera_d, focal=self.focal, center=center,
                is_train=True, 
                opt_id_dim = 526,
                opt_exp_dim = 203,
                opt_tex_dim = 80,
                use_region_uv = False,
                used_region_tex_type = ['uv'],
                use_external_exp = False
            )
        self.renderer = MeshRenderer(
            rasterize_fov=self.fov, znear=1, zfar=20, rasterize_size=self.resolution
        )
        self.texRes = texture_resolution
        self.dpRes = dp_map_resolution
        self.device=device
        self.latent_init=latent_init
        self.dp_map_scale=dp_map_scale
        self.init_parameters()
        self.set_transformation_range()
        
        with torch.no_grad():
            if self.stage != 'coarse geometry generation':
                self.load_shape(saved_id_path,saved_dp_path)
            if self.stage == 'edit':
                self.load_diffuse(saved_diffuse_path)

        self.define_optim_param()


    # load functions
    from models.stage_fitter._forward import forward, render_specific_view, generate_diffuse, encode_diffuse_latent, compute_facial_mask,bi_direction_projection,render_latent,render_control_img
    from models.stage_fitter._render_util  import random_transformation, apply_transformation,compute_transformed_vertex, random_light, set_gamma, normalize_depth, normalize_depth_with_camerad
    from models.stage_fitter._shape_util import predict_dp, update_shape,shape2posmap
    from models.stage_fitter._io_util import load_shape, save_visuals, save_results,render_uv_from_mlp,load_diffuse,save_attention


    def init_parameters(self):
        self.id_para = nn.Parameter(torch.zeros(1, 526).float().to(self.device))
        self.exp_para = nn.Parameter(torch.zeros(1, 203).float().to(self.device))
        self.tex_para = nn.Parameter(torch.zeros(1, 80).float().to(self.device))
        self.angles_para = nn.Parameter(torch.zeros(1, 3).float().to(self.device))
        self.trans_para = nn.Parameter(torch.zeros(1, 3).float().to(self.device))
        self.gamma_para = nn.Parameter(torch.tensor([0.6,-0.2,0.25,-0.15,0,-0.15,0,0,0] * 3)[None, ...].float().to(self.device))
        self.diffuse_texture = nn.Parameter(torch.ones(1,self.texRes,self.texRes,3).float().to(self.device) * 0.5)
        if self.latent_init == 'randn':
            self.diffuse_latent = nn.Parameter(torch.randn(1,4,64,64).float().to(self.device))
        elif self.latent_init == 'ones':
            self.diffuse_latent = nn.Parameter(torch.ones(1,4,64,64).float().to(self.device))
        elif self.latent_init=='zeros':
            self.diffuse_latent = nn.Parameter(torch.zeros(1,4,64,64).float().to(self.device))
        self.dp_tensor = nn.Parameter(torch.zeros(1,self.dpRes,self.dpRes,1).float().to(self.device))

        ### not optimizable parameters
        self.constant_diffuse_texture = torch.ones(1,self.texRes,self.texRes,3).float().to(self.device)*0.5
        self.rotation=[0,0,0]
        self.translation_z=0        
        self.lights = random_gamma(self.device,strength=0.2)
        self.pred_vertex_no_pose = torch.zeros(1, 20481, 3).float().to(self.device)


    def define_optim_param(self):
        self.optim_param = []
        if self.stage == 'coarse geometry generation':
            self.optim_param = [self.id_para, self.diffuse_texture, self.diffuse_latent]
        if self.stage == 'texture generation':
            if self.diffuse_generation_type == 'latent':
                self.optim_param = [self.diffuse_latent]
            elif self.diffuse_generation_type == 'direct':
                self.optim_param = [self.diffuse_texture]

        if self.stage == 'edit':
            if self.edit_scope == 'tex':
                self.optim_param = [self.diffuse_latent]
            elif self.edit_scope == 'geo':
                self.optim_param = [self.id_para]
               
    def get_parameters(self):
        if self.stage in ['texture generation','edit'] and self.diffuse_generation_type == 'mlp':
            return self.diffuse_mlp.parameters()
        else:
            return self.optim_param
        
    def to(self,device):
        self.facemodel.to(device)
        if self.stage in ['texture generation','edit'] and self.diffuse_generation_type == 'mlp':
            self.diffuse_mlp.to_(device)
        else:
            for param in self.get_parameters():
                param = param.to(device)
    
    def set_transformation_range(self,x_min_max=[0,0],y_min_max=[0,0],z_min_max=[0,0],t_z_min_max=[0,0]):
        assert x_min_max[0] <= x_min_max[1] and y_min_max[0] <= y_min_max[1] and z_min_max[0] <= z_min_max[1] and t_z_min_max[0] <= t_z_min_max[1]
        self.x_min_max = x_min_max
        self.y_min_max = y_min_max
        self.z_min_max = z_min_max
        self.t_z_min_max = t_z_min_max

    def set_transformation_choices(self,x_list,y_list,z_list,t_z_list):
        self.x_list = x_list
        self.y_list = y_list
        self.z_list = z_list
        self.t_z_list = t_z_list

    def concat_coeff(self):
        ID_para = self.id_para
        EXP_para = self.exp_para
        TEX_para = self.tex_para
        ANGLES = self.angles_para
        GAMMAS = self.gamma_para
        TRANSLATIONS = self.trans_para
        return torch.cat([ID_para,EXP_para,TEX_para,ANGLES,GAMMAS,TRANSLATIONS],dim=1)
    
    def set_texture_from_img(self,img_path):
        from PIL import Image
        I = torch.tensor(np.array(Image.open(img_path,'r').resize([512,512]).convert('RGB'))/255.).float().to(self.device).unsqueeze(0)
        self.diffuse_texture = I

    def set_shape_from_file(self,coeff_path,dp_path=None):
        self.load_shape(coeff_path,dp_path)
        self.update_shape()

    def set_shape_from_obj(self,obj_path):
        from util import meshio
        obj = meshio.Mesh(obj_path)
        recon_shape = torch.tensor(obj.vertices).float().unsqueeze(0)
        recon_shape[..., -1] = self.camera_d - recon_shape[0][..., -1] # from camera space to world space
        self.pred_vertex_no_pose = recon_shape.to(self.device)
        self.pred_vertex = self.pred_vertex_no_pose.clone()
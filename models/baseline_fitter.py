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

class BaselineFitter(object):
    def __init__(self,render_resolution=224,fov=12.593637,camera_d=10,
                    texture_resolution=512, dp_map_resolution=128,
                device='cuda',fit_param=['pose','shape','tex','light'],
                fixed_id_path = None,
                fixed_dp_map_path = None,
                latent_init='zeros',
                dp_map_scale=0.0025
                ):

        # camera setting
        self.resolution = render_resolution
        self.fov = fov
        self.camera_d = camera_d
        center = self.resolution / 2
        self.focal = center / np.tan(self.fov * np.pi / 360)
        
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
            rasterize_fov=self.fov, znear=5, zfar=15, rasterize_size=self.resolution
        )
        self.texRes = texture_resolution
        self.dpRes = dp_map_resolution
        self.device=device
        self.fit_param=fit_param
        self.latent_init=latent_init
        self.dp_map_scale=dp_map_scale
        self.init_parameters()
        self.set_transformation_range()

        if fixed_id_path:
            self.load_shape(fixed_id_path,fixed_dp_map_path)

    def init_parameters(self):
        self.id_para = nn.Parameter(torch.zeros(1, 526).float().to(self.device))
        self.exp_para = nn.Parameter(torch.zeros(1, 203).float().to(self.device))
        self.tex_para = nn.Parameter(torch.zeros(1, 80).float().to(self.device))
        self.angles_para = nn.Parameter(torch.zeros(1, 3).float().to(self.device))
        self.trans_para = nn.Parameter(torch.zeros(1, 3).float().to(self.device))
        # self.gamma_para = nn.Parameter(torch.ones(1, 27).float().to(self.device) * 0.5)
        self.gamma_para = nn.Parameter(torch.tensor([0.6,-0.2,0.25,-0.15,0,-0.15,0,0,0] * 3)[None, ...].float().to(self.device))
        self.diffuse_texture = nn.Parameter(torch.ones(1,self.texRes,self.texRes,3).float().to(self.device) * 0.5)

        if self.latent_init == 'randn':
            self.diffuse_latent = nn.Parameter(torch.randn(1,4,64,64).float().to(self.device))
        elif self.latent_init == 'ones':
            self.diffuse_latent = nn.Parameter(torch.ones(1,4,64,64).float().to(self.device))
        elif self.latent_init=='zeros':
            self.diffuse_latent = nn.Parameter(torch.zeros(1,4,64,64).float().to(self.device))

        self.dp_tensor = nn.Parameter(torch.zeros(1,self.dpRes,self.dpRes,1).float().to(self.device))
        # self.diffuse_latent = nn.Parameter(torch.randn(1,4,64,64).float().to(self.device) * 0.18215)



        ### not optimizable parameters
        self.constant_diffuse_texture = torch.ones(1,self.texRes,self.texRes,3).float().to(self.device)*0.5
        self.rotation=[0,0,0]
        self.translation_z=0        
        self.lights = random_gamma(self.device)
        self.pred_vertex_no_pose = torch.zeros(1, 20481, 3).float().to(self.device)
        
        # self.diffuse_texture = nn.Parameter(torch.rand(1,self.texRes,self.texRes,3).float().to(self.device))

        self.optim_param = []
        if 'pose' in self.fit_param:
            self.optim_param += [self.angles_para,self.trans_para]
        if 'id' in self.fit_param:
            self.optim_param += [self.id_para]
        if 'exp' in self.fit_param:
            self.optim_param += [self.exp_para]
        if 'tex' in self.fit_param:
            self.optim_param += [self.diffuse_texture]
        if 'light' in self.fit_param:
            self.optim_param += [self.gamma_para]
        if 'latent' in self.fit_param:
            self.optim_param += [self.diffuse_latent]
        if 'dp' in self.fit_param:
            self.optim_param += [self.dp_tensor]


    def predict_dp(self):
        self.dp_map = F.tanh(self.dp_tensor) * self.dp_map_scale

    def load_shape(self,coeff_path,dp_path=None):
        # mesh = meshio.Mesh(obj_path)
        # self.pred_vertex_no_pose = torch.tensor(mesh.vertices).float().to(self.device)
        # self.pred_vertex_no_pose[..., -1] = self.camera_d - self.pred_vertex_no_pose[..., -1] # from camera space to world space
        with torch.no_grad():
            self.id_para[:] = torch.tensor(np.load(coeff_path)).float().to(self.device)[:]
            self.coarse_id_para = torch.tensor(np.load(coeff_path)).float().to(self.device).clone()
            if dp_path != None:
                self.dp_tensor = torch.tensor(np.load(dp_path)).float().to(self.device)[:]
        
    def get_parameters(self):
        return self.optim_param#[self.id_para,self.exp_para,self.tex_para,self.angles_para,self.trans_para,self.gamma_para,self.diffuse_texture]

    def to(self,device):
        self.facemodel.to(device)
        for param in self.get_parameters():
            param = param.to(device)
    
    def set_transformation_range(self,x_min_max=[0,0],y_min_max=[0,0],z_min_max=[0,0],t_z_min_max=[0,0]):
        assert x_min_max[0] <= x_min_max[1] and y_min_max[0] <= y_min_max[1] and z_min_max[0] <= z_min_max[1] and t_z_min_max[0] <= t_z_min_max[1]
        self.x_min_max = x_min_max
        self.y_min_max = y_min_max
        self.z_min_max = z_min_max
        self.t_z_min_max = t_z_min_max

    def random_transformation(self):
        # rotation and translation-z
        return [random.uniform(self.x_min_max[0],self.x_min_max[1]),
                random.uniform(self.y_min_max[0],self.y_min_max[1]),
                random.uniform(self.z_min_max[0],self.z_min_max[1]),],   random.uniform(self.t_z_min_max[0],self.t_z_min_max[1])

    def apply_transformation(self,rotatation=[0,0,0],translation_z=0):
        '''
        transform the face with rotation, translation, while keeps the face in the center of the rendered
        rotation: triple array, [x,y,z], x: up-down, y:left-right, z:default 0
        translation_z: to be test
        '''
        # first translate to the original point
        center = torch.mean(self.pred_vertex,dim=1)
        vertex_transformed_to_origin = self.pred_vertex - center
        
        # then rotate
        rotation = self.facemodel.compute_rotation(degree2radian(torch.tensor(rotatation).view(1,3).float()).to(self.facemodel.device))
        
        # translate back to center
        vertex_rotated = self.facemodel.transform(vertex_transformed_to_origin, rotation, center)
        
        # trnaslate the translation_z
        vertex_final = vertex_rotated + torch.tensor([0,0,translation_z]).view(1,3).float().to(vertex_rotated.device)
        self.pred_vertex = vertex_final

    def random_light(self):
        self.gamma_para = self.lights[random.randint(0,len(self.lights)-1)]

    def concat_coeff(self):
        ID_para = self.id_para
        EXP_para = self.exp_para
        TEX_para = self.tex_para
        ANGLES = self.angles_para
        GAMMAS = self.gamma_para
        TRANSLATIONS = self.trans_para
        return torch.cat([ID_para,EXP_para,TEX_para,ANGLES,GAMMAS,TRANSLATIONS],dim=1)
    
    def update_shape(self):
        output_coeff = self.concat_coeff()
        self.predict_dp()
        if 'dp' in self.fit_param:
            self.pred_vertex, self.pred_vertex_norm = \
                self.facemodel.compute_for_render_with_dp_map(output_coeff,self.dp_map,use_external_exp=False)    # print(self.pred_vertex.shape) # torch.Size([1, 20481, 3])
        else:
            self.pred_vertex, self.texture_3dmm, self.pred_vertex_norm, self.pred_lm = \
                self.facemodel.compute_for_render(output_coeff,use_external_exp=False)    # print(self.pred_vertex.shape) # torch.Size([1, 20481, 3])
        self.pred_vertex_no_pose = self.pred_vertex.detach().clone()


    def forward(self,diffuse_type='constant', input_diffuse = None,
                    update_view=True,employ_random_light=True):

        output_coeff = self.concat_coeff()

        self.update_shape()
        
        # 随机采一个新视角
        if update_view:
            self.rotation,self.translation_z = self.random_transformation()

        # 应用视角变换
        self.apply_transformation(self.rotation,self.translation_z)        

        # 随机采用一个新的光照
        if employ_random_light:
            self.random_light()
        
        # 使用白模渲染
        if diffuse_type == 'constant':
            self.pred_mask, self.pred_depth, pix_tex, pix_norm = self.renderer.forward_with_texture_map(
                    self.pred_vertex, self.facemodel.face_buf, self.constant_diffuse_texture, self.facemodel.vt, self.facemodel.face_vt, self.pred_vertex_norm)
        # 使用diffuse渲染
        elif diffuse_type == 'diffuse':
            self.pred_mask, self.pred_depth, pix_tex, pix_norm = self.renderer.forward_with_texture_map(
                    self.pred_vertex, self.facemodel.face_buf, self.diffuse_texture, self.facemodel.vt, self.facemodel.face_vt, self.pred_vertex_norm)            

        # 渲染法向图
        pix_norm = F.normalize(pix_norm,dim=1)
        self.pix_tex = pix_tex
        self.pix_norm = pix_norm
        self.pred_face = self.facemodel.compute_color(pix_tex.permute(0,2,3,1),pix_norm.permute(0,2,3,1),output_coeff,white_light=False) #pred_face:[bs,imgsize,imgsize,3]
        self.pred_face = self.pred_face.contiguous().permute(0,3,1,2)
        return self.pred_face, self.pix_tex, self.pred_depth, self.pix_norm


    def full_stage_forward(self,stage):
        assert stage == 'coarse gometry generation' or stage == 'texture generation'

        if stage == 'texture generation':# 如果是贴图预测阶段，则几何使用固定的几何
            self.pred_vertex = self.pred_vertex_no_pose.detach().clone()
        else:# 如果是几何预测阶段，则需要优化几何
            output_coeff = self.concat_coeff()
            self.update_shape()
        
        
        # 随机采一个新视角
        self.rotation,self.translation_z = self.random_transformation()
        self.apply_transformation(self.rotation,self.translation_z)
        # 随机采用一个新的光照
        self.random_light()

        # 计算全灰贴图的光栅化结果
        _, _, grey_pix_tex, _ = self.renderer.forward_with_texture_map(
                    self.pred_vertex, self.facemodel.face_buf, self.constant_diffuse_texture, self.facemodel.vt, self.facemodel.face_vt, self.pred_vertex_norm)
        # 计算diffuse贴图的光栅化结果
        self.pred_mask, self.pred_depth, pix_tex, pix_norm = self.renderer.forward_with_texture_map(
                    self.pred_vertex, self.facemodel.face_buf, self.diffuse_texture, self.facemodel.vt, self.facemodel.face_vt, self.pred_vertex_norm)

        pix_norm = F.normalize(pix_norm,dim=1)
        self.grey_pix_tex = grey_pix_tex
        self.pix_tex = pix_tex
        self.pix_norm = pix_norm
        # 渲染全灰白模结果
        self.grey_pred_face = self.facemodel.compute_color(grey_pix_tex.permute(0,2,3,1),pix_norm.permute(0,2,3,1),output_coeff,white_light=False) #pred_face:[bs,imgsize,imgsize,3]
        self.grey_pred_face = self.grey_pred_face.contiguous().permute(0,3,1,2)

        # 渲染diffuse结果
        self.pred_face = self.facemodel.compute_color(pix_tex.permute(0,2,3,1),pix_norm.permute(0,2,3,1),output_coeff,white_light=False) #pred_face:[bs,imgsize,imgsize,3]
        self.pred_face = self.pred_face.contiguous().permute(0,3,1,2)

        return self.pred_face, self.grey_pred_face, self.pred_depth, self.pix_norm
    
        


    def set_gamma(self,gamma):
        self.gamma_para = gamma

    def render_specific_view(self,rotation,translation_z,input_gamma=None):
        with torch.no_grad():
            # 设置光照
            if input_gamma!=None:
                self.set_gamma(input_gamma)
            else:
                self.set_gamma(torch.tensor([0.6,-0.2,0.25,-0.15,0,-0.15,0,0,0] * 3)[None, ...].float().to(self.device))

            output_coeff = self.concat_coeff()
            # self.pred_vertex, self.texture_3dmm, self.pred_vertex_norm, self.pred_lm = \
            #         self.facemodel.compute_for_render(output_coeff,use_external_exp=False)
            self.pred_vertex = self.pred_vertex_no_pose.detach().clone()
            
            self.pred_vertex_norm = self.facemodel.compute_norm(self.facemodel.to_world(self.pred_vertex_no_pose.clone()))  # 相机坐标下的旋转，不影响世界坐标下的法线方向

            # self.pred_vertex_, self.texture_3dmm, self.pred_vertex_norm_, self.pred_lm = \
            #         self.facemodel.compute_for_render(output_coeff,use_external_exp=False)


            self.apply_transformation(rotation,translation_z)      

            # grey tex
            pred_mask, pred_depth, pix_grey_tex, pix_norm = self.renderer.forward_with_texture_map(
                    self.pred_vertex, self.facemodel.face_buf, self.constant_diffuse_texture, self.facemodel.vt, self.facemodel.face_vt, self.pred_vertex_norm)            
            # diffuse tex
            pred_mask, pred_depth, pix_tex, pix_norm = self.renderer.forward_with_texture_map(
                    self.pred_vertex, self.facemodel.face_buf, self.diffuse_texture, self.facemodel.vt, self.facemodel.face_vt, self.pred_vertex_norm)

            pix_norm = F.normalize(pix_norm,dim=1)

            # grey face
            pred_grey_face = self.facemodel.compute_color(pix_grey_tex.permute(0,2,3,1),pix_norm.permute(0,2,3,1),output_coeff,white_light=False) #pred_face:[bs,imgsize,imgsize,3]
            pred_grey_face = pred_grey_face.contiguous().permute(0,3,1,2)
            pred_face = self.facemodel.compute_color(pix_tex.permute(0,2,3,1),pix_norm.permute(0,2,3,1),output_coeff,white_light=False) #pred_face:[bs,imgsize,imgsize,3]
            pred_face = pred_face.contiguous().permute(0,3,1,2)

            return pred_face, pred_grey_face, pix_norm

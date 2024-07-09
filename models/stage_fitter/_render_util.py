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

def random_transformation(self, with_choice=False):
    '''
    with_choice:
    True: Select only a fixed number of angles
    False: randomly generate an Angle
    '''
    # rotation and translation-z
    if with_choice:
        return [random.choice(self.x_list),
                random.choice(self.y_list),
                random.choice(self.z_list)],   random.choice(self.t_z_list)
    else:
        return [random.uniform(self.x_min_max[0],self.x_min_max[1]),
                random.uniform(self.y_min_max[0],self.y_min_max[1]),
                random.uniform(self.z_min_max[0],self.z_min_max[1]),],   random.uniform(self.t_z_min_max[0],self.t_z_min_max[1])


def compute_transformed_vertex(self,input_vertex, rotatation=[0,0,0],translation_z=0):
    '''
    transform the face with rotation, translation, while keeps the face in the center of the rendered
    rotation: triple array, [x,y,z], x: up-down, y:left-right, z:default 0
    translation_z: to be test
    '''
    # first translate to the original point
    center = torch.mean(input_vertex,dim=1)
    vertex_transformed_to_origin = input_vertex - center
    
    # then rotate
    rotation = self.facemodel.compute_rotation(degree2radian(torch.tensor(rotatation).view(1,3).float()).to(self.facemodel.device))
    
    # translate back to center
    vertex_rotated = self.facemodel.transform(vertex_transformed_to_origin, rotation, center)
    
    # trnaslate the translation_z
    vertex_final = vertex_rotated + torch.tensor([0,0,translation_z]).view(1,3).float().to(vertex_rotated.device)
    return vertex_final

def apply_transformation(self,rotatation=[0,0,0],translation_z=0):
    '''
    transform the face with rotation, translation, while keeps the face in the center of the rendered
    rotation: triple array, [x,y,z], x: up-down, y:left-right, z:default 0
    translation_z: to be test
    '''
    self.pred_vertex = self.compute_transformed_vertex(self.pred_vertex, rotatation,translation_z)

def random_light(self):
    self.gamma_para = self.lights[random.randint(0,len(self.lights)-1)]


def set_gamma(self,gamma):
    self.gamma_para = gamma

def normalize_depth(self, depth_map):
    object_mask = depth_map != 0

    min_val = 0.5
    depth_map[object_mask] = ((1 - min_val) * (depth_map[object_mask] - depth_map[object_mask].min()) / (
        depth_map[object_mask].max() - depth_map[object_mask].min())) + min_val

    return depth_map

def normalize_depth_with_camerad(self,depth_map):
    object_mask = depth_map != 0

    min_val = 0.5
    depth_map[object_mask] = depth_map[object_mask] - self.camera_d

    depth_map[object_mask] = ((1 - min_val) * (depth_map[object_mask] - depth_map[object_mask].min()) / (
        depth_map[object_mask].max() - depth_map[object_mask].min())) + min_val
    # Implicates max_val=1
    return depth_map
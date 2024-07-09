"""This script contains basic utilities for Deep3DFaceRecon_pytorch
"""
from __future__ import print_function
import numpy as np
import torch
from PIL import Image
import os
import importlib
import argparse
from argparse import Namespace
import torchvision
import time

import yaml
import json
# from lib.Deep3DHIFI.models.losses import gaussian_blur
# from lib.Deep3DHIFI.lib.FFHQUV.tex.tex_func import match_color_in_yuv
from typing import List
import pprint


# def match_template(template,uwp,msk):
#     src_tex = uwp.detach().cpu().numpy() * 255.
#     msk = msk.detach().cpu().numpy() * 255.
#     matched_template = match_color_in_yuv(template,src_tex,msk)
#     return matched_template

# def fuse_by_msk(uv1,uv2,msk,kernel_factor=32):
#     '''
#     uv1,uv2,msk : BxCxWxH
#     '''
#     blur_msk = gaussian_blur(msk, msk.shape[2] // kernel_factor + 1)
#     return uv1*blur_msk + uv2*(1-blur_msk)

def yaml2json(y_file,j_file):
    with open(y_file, 'r') as file:
        configuration = yaml.safe_load(file)

    with open(j_file, 'w') as json_file:
        json.dump(configuration, json_file)
        
    output = json.dumps(json.load(open(j_file)), indent=2)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def timing(prefix, last):
    torch.cuda.synchronize()
    tmp = time.time()
    print(f'{prefix} costs {tmp-last} s')
    return tmp

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf

def genvalconf(train_opt, **kwargs):
    conf = Namespace(**vars(train_opt))
    attr_dict = train_opt.__dict__
    for key, value in attr_dict.items():
        if 'val' in key and key.split('_')[0] in attr_dict:
            setattr(conf, key.split('_')[0], value)

    for key in kwargs:
        setattr(conf, key, kwargs[key])

    return conf
        
def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array, range(0, 1)
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.clamp(0.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def correct_resize_label(t, size):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i, :1]
        one_np = np.transpose(one_t.numpy().astype(np.uint8), (1, 2, 0))
        one_np = one_np[:, :, 0]
        one_image = Image.fromarray(one_np).resize(size, Image.NEAREST)
        resized_t = torch.from_numpy(np.array(one_image)).long()
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


def correct_resize(t, size, mode=Image.BICUBIC):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i:i + 1]
        one_image = Image.fromarray(tensor2im(one_t)).resize(size, Image.BICUBIC)
        resized_t = torchvision.transforms.functional.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)

def draw_landmarks(img, landmark, color='r', step=2):
    """
    Return:
        img              -- numpy.array, (B, H, W, 3) img with landmark, RGB order, range (0, 255)
        

    Parameters:
        img              -- numpy.array, (B, H, W, 3), RGB order, range (0, 255)
        landmark         -- numpy.array, (B, 68, 2), y direction is opposite to v direction
        color            -- str, 'r' or 'b' (red or blue)
    """
    if color =='r':
        c = np.array([255., 0, 0])
    else:
        c = np.array([0, 0, 255.])

    _, H, W, _ = img.shape
    img, landmark = img.copy(), landmark.copy()
    landmark[..., 1] = H - 1 - landmark[..., 1]
    landmark = np.round(landmark).astype(np.int32)
    for i in range(landmark.shape[1]):
        x, y = landmark[:, i, 0], landmark[:, i, 1]
        for j in range(-step, step):
            for k in range(-step, step):
                u = np.clip(x + j, 0, W - 1)
                v = np.clip(y + k, 0, H - 1)
                for m in range(landmark.shape[0]):
                    img[m, v[m], u[m]] = c
    return img


def load_opt(json_path):
    import argparse,json
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    opt, _ = parser.parse_known_args()
    with open(json_path, 'r') as json_file:
        opt.__dict__ = json.load(json_file)
    return opt

def opt_stage_table():
    return {
        1:['shape_params','expression_params','tex_params','white_light'],
        2:['pred_normal','highRes_coarse_tex','flistHighResTex','flistHighResTex_val','tex_resolution','flist','flist_val','flistImg','flistImg_val','flistUnwrapped','flistUnwrapped_val'],
        3:['lightnet_type','shade_type','brdf_type'],
        4:[],
        5:[],
    }

def write_obj(obj_path, v_arr, vt_arr, tri_v_arr, tri_t_arr, tex_filepath=None):
    """write mesh data to .obj file.

    Param:
    obj_path: path to .obj file
    v_arr   : N x 3 (x, y, z values for geometry)
    vt_arr  : N x 2 (u, v values for texture)
    f_arr   : M x 3 (mesh faces and their corresponding triplets)

    Returns:
    None
    """
    tri_v_arr = np.copy(tri_v_arr)
    tri_t_arr = np.copy(tri_t_arr)
    mtl_name = os.path.basename(obj_path)[:-4] + '.mtl'
    mtl_path = obj_path[:-4]+'.mtl'
    if tex_filepath != None:
        tex_filename = os.path.basename(tex_filepath)
        with open(mtl_path,'w') as fp:
            fp.write('newmtl test\n')
            fp.write('Ka 0.2 0.2 0.2\n')
            fp.write('Kd 0.8 0.8 0.8\n')
            fp.write('Ks 1.0 1.0 1.0\n')
            fp.write('map_Kd '+ tex_filename+'\n')
    with open(obj_path, "w") as fp:
        fp.write('mtllib '+mtl_name+'\n')
        for x, y, z in v_arr:
            fp.write("v %f %f %f\n" % (x, y, z))
        # for u, v in vt_arr:
        #    fp.write('vt %f %f\n' % (v, 1-u))
        tri_v_arr += 1
        if type(vt_arr) != type(None) and type(tri_t_arr) != type(None):
            tri_t_arr += 1
            if np.amax(vt_arr) > 1 or np.amin(vt_arr) < 0:
                print("Error: the uv values should be ranged between 0 and 1")
            for u, v in vt_arr:
                fp.write("vt %f %f\n" % (u, v))


            for (v1, v2, v3), (t1, t2, t3) in zip(tri_v_arr, tri_t_arr):
                fp.write(
                    "f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (v1, t1, t1, v2, t2, t2, v3, t3, t3)
                )
        else:
            for (v1,v2,v3) in tri_v_arr:
                fp.write(
                    "f %d %d %d\n" % (v1,  v2, v3)
                )

from torch.cuda.amp import custom_bwd, custom_fwd
class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)


def mean_list(valid_ids, valid_values, stride=2, iters=1000):
    length = len(valid_ids)
    nums = length // stride
    used_ids = []
    mean_values = []
    std_vallues = []
    valid_values = np.asarray(valid_values)
    for k in range(nums):
        used_ids.append(int(k*stride*iters))
        temp_list = valid_values[k*stride:(k+1)*stride]
        mean = np.mean(temp_list)
        std = np.std(temp_list)
        mean_values.append(mean)
        std_vallues.append(std)
    return used_ids, mean_values, std_vallues

def list2str(l):
    s = ''
    for i in l:
        s+=i
    return s

def SD_decode_latents(SD, latents):
    SD.vae.requires_grad_(False)
    latents = 1 / 0.18215 * latents

    imgs = SD.vae.decode(latents).sample

    imgs = (imgs / 2 + 0.5).clamp(0, 1)

    return imgs

def SD_encode_imgs(SD,imgs):
    SD.vae.requires_grad_(False)
    imgs = 2 * imgs - 1
    posterior = SD.vae.encode(imgs).latent_dist
    latents = posterior.sample() * 0.18215
    return latents


def get_indices_to_alter(stable, prompt: str, indices_to_alter_str: str = "") -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                            for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                            if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    if indices_to_alter_str == "":  # 手动输入
        token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                            "alter (e.g., 2,5), 0 indicates no attention-constraint employed: ")
    else:
        token_indices = indices_to_alter_str
    token_indices = [int(i) for i in token_indices.split(",")]
    if 0 in token_indices:
        return None
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices
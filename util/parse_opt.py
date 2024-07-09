import argparse
from util import util 
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser()
    # exp setttings
    parser.add_argument('--device', type=str,default='cuda')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--total_steps', type=int, default=201,help='total optimize steps')
    parser.add_argument('--save_freq', type=int, default=20,help='freq for saving optimized results')
    parser.add_argument('--exp_root', type=str,default='exp',help='result save root')
    parser.add_argument('--exp_name', type=str,default='mul-view',help='result save folder name')
    
    parser.add_argument('--path_debug', type=util.str2bool,default=False,help='if true, the exp_name will append the exp setting, if false, use exp_name directly')

    parser.add_argument("--fit_param", type=str,default = 'id tex',help='params to be optimized, ["id", "tex", "pose", "exp", "light"]')
    parser.add_argument("--lr", type=float,default = 0.05,help='the learning rate')
    
    parser.add_argument("--stage",type=str,default='coarse geometry generation',choices=['coarse geometry generation', 'texture generation','edit'])

    # render settings
    parser.add_argument('--render_resolution', type=int,default=224)
    
    # viewpoint settings
    parser.add_argument('--viewpoint_range_X_min', type=float,default=-20,help='define the random viewpoint rotation range in optimization')
    parser.add_argument('--viewpoint_range_X_max', type=float,default=20,help='define the random viewpoint rotation range in optimization')
    parser.add_argument('--viewpoint_range_Y_min', type=float,default=-45,help='define the random viewpoint rotation range in optimization')
    parser.add_argument('--viewpoint_range_Y_max', type=float,default=45,help='define the random viewpoint rotation range in optimization')
    parser.add_argument('--viewpoint_range_Z_min', type=float,default=0,help='define the random viewpoint rotation range in optimization')
    parser.add_argument('--viewpoint_range_Z_max', type=float,default=0,help='define the random viewpoint rotation range in optimization')

    parser.add_argument('--force_fixed_viewpoint',type=util.str2bool,default=True,help='fix some certain viewpoints for texture generation/edit')

    parser.add_argument('--t_z_min', type=float,default=0,help='define the random viewpoint translation range in optimization')
    parser.add_argument('--t_z_max', type=float,default=3,help='define the random viewpoint translation range in optimization')


    parser.add_argument('--display_rotation_x', type=float,default=10,help='define the fixed viewpoint rotation for visualization')
    parser.add_argument('--display_rotation_y', type=float,default=10,help='define the fixed viewpoint rotation for visualization')
    parser.add_argument('--display_rotation_z', type=float,default=0,help='define the fixed viewpoint rotation for visualization')
    parser.add_argument('--display_translation_z', type=float,default=1.5,help='define the fixed viewpoint translation for visualization')

    # dp map settings
    parser.add_argument('--dp_map_scale',type=float,default=0.0025)

    # texture generation settings
    parser.add_argument("--texture_generation",type=str, default='latent',choices=['direct','latent','mlp'],help='the way to generation the diffuse texture')
    parser.add_argument("--latent_init",type=str,default='zeros',choices=['randn','ones','zeros'])
    parser.add_argument("--textureLDM_path",type=str,default="./ckpts/TextureDiffusion/unet")

    # edit settings
    parser.add_argument('--edit_prompt_cfg',type=float,default=100)
    parser.add_argument('--edit_img_cfg',type=float,default=20)
    parser.add_argument('--edit_scope',type=str,choices=['tex','geo'],default='tex')


    # guidance settings
    parser.add_argument('--guidance_type', type=str,default='stable-diffusion',choices=['stable-diffusion','clip'],help='the guidance type for text prompt')
    parser.add_argument('--sd_version', type=str,default='2.1',choices=['1.5','2.0','2.1'],help='the stable-diffusion version for sds loss')
    parser.add_argument('--controlnet_name', type=str, choices = ['depth','normal'] ,default='depth')
    parser.add_argument('--vis_att',type=util.str2bool,default=False)


    # prompt settings
    parser.add_argument('--text', type=str,required=True,default=None)
    parser.add_argument('--negative_text', type=str,default='')
    parser.add_argument('--use_view_adjust_prompt',type=util.str2bool,default=True)
    # static text for textureLDM
    parser.add_argument('--static_text', type=str,default='a diffuse texture map of a human face in UV space')
    parser.add_argument('--use_static_text', type=util.str2bool,default=True,help='if indicated, the textureLDM will use the static_text rather than the text param')

    # sds loss rendering settings
    parser.add_argument("--sds_input",type=str, default='rendered',help='rendered type to be sent into SD for loss, this parameter is available only in the coarse geometry generation or edit geo generation')
    parser.add_argument("--random_light",type=util.str2bool, default=True,help='whether sample random light in rendering for loss')
    parser.add_argument("--w_SD",type=float,default=1.0,help='weight for sds loss from stable diffusion')
    parser.add_argument("--w_texSD",type=float,default=3.0,help='weight for sds loss from texture LDM')
    parser.add_argument("--cfg_SD",type=float,default=100,help='class-free guidance strength for SD')
    parser.add_argument("--cfg_texSD",type=float,default=1,help='class-free guidance strength for texture LDM')
    parser.add_argument("--set_t_schedule",type=util.str2bool,default=True,help='whether set the timestep in SDS as a normal decreasing schedule in denoise process. False: uniform sampling, True, decreasing with equal interval')
    parser.add_argument("--schedule_type",type=str,default='linear',choices=['linear','uniform_rand','dreamtime'])
    parser.add_argument("--set_w_schedule",type=util.str2bool,default=False,help='whether set the weight of texSD in SDS as a decreasing schedule')
    parser.add_argument("--w_schedule",type=str,choices=['linear','log'],default='linear',help='the schedule way of w_texSD')
    parser.add_argument("--w_texSD_max",type=float,default=20,help='the max value of w_texSD in schedule')
    parser.add_argument("--w_texSD_min",type=float,default=3,help='the min value of w_texSD in schedule')
    parser.add_argument('--latent_sds_steps', type=int, default=201,help='the steps of latent sds')
    
    # TextureLDM in YUV space settings
    parser.add_argument("--employ_yuv",type=util.str2bool,default=False,help='whether use yuv textureLDM')
    parser.add_argument("--textureLDM_yuv_path",type=str,default="./ckpts/TextureDiffusion-yuv/unet")
    parser.add_argument("--w_texYuv",type=float,default=1)

    # edit regularization loss
    parser.add_argument('--w_reg_diffuse',type=float,default=1,help='the reg of diffuse consistency before and after editing')
    parser.add_argument('--attention_reg_diffuse',type=util.str2bool,default=False,help='whether employ attention map from SD to weight the reg diffuse')
    parser.add_argument('--attention_sds',type=util.str2bool,default=False,help='whether employ attention map from SD to weight the pix2pix sds gradient')

    parser.add_argument('--scp_fuse',type=str,default='avm2',choices=['avg0.5','avg2','max2','avm2'],help='the way to fuse scp msk in different iteration\
                                                                                                        avg0.5: res = w * res + (1-w) * (1-tmp**0.5)\
                                                                                                        avg2:   res = w * res + (1-w)*(1-tmp**2) \
                                                                                                        max2:   tmp_s = 1-tmp**2 \
                                                                                                                res = (tmp_s>res) * tmp + (tmp_s<=res) * res \
                                                                                                        avm2:   tmp_s = 1-tmp**2 \
                                                                                                                res = (tmp_s>res) * (tmp * (1-w) + res * w) + \
                                                                                                                      (tmp_s<=res) * res'
                                                                                                                )

    parser.add_argument('--indices_to_alter_str',type=str,default="")

    # texture regularization loss
    parser.add_argument('--w_sym',type=float,default=0)
    parser.add_argument('--w_smooth',type=float,default=0)

    # pre-computed mesh settings
    parser.add_argument('--load_id_path',type=str,default=None)
    parser.add_argument('--load_dp_path',type=str,default=None)
    parser.add_argument('--load_diffuse_path',type=str,default=None)


    opt = parser.parse_args()
    opt.fit_param = opt.fit_param.split()
    opt.sds_input = opt.sds_input.split()

    # print(opt)

    return opt
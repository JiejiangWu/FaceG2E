import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import os
from util.parse_opt import parse_args
from util import util
from models.stage_fitter import StageFitter
from models.sd import StableDiffusion
from models.sd_textureLDM import StableDiffusion_w_texture
from models.instructp2p import StableDiffusion_instructp2p
from models import CLIP
from tqdm import tqdm
import random
import json
from torch.optim.lr_scheduler import StepLR
from kornia.color import RgbToYuv
from util.T_scheduler import T_scheduler
from util.prompt_util import prompt_suffix
import pathlib
from util.loss import attention_mask, texture_symmetric_loss,texture_smooth_loss

def train_step(fitter, prompt, text_z, static_text_z, opt,
                sds_input=['rendered'],employ_textureLDM=False,iter_step=0, total_steps=200,attention_store=None,indices_to_alter=None):

    # 随机一种渲染形式作为SDS的输入    
    dim = len(sds_input)
    select_input_type = sds_input[random.randint(0,dim-1)]


    # 如果是几何阶段
    if fitter.stage in ['coarse geometry generation']:
        # 进行forward，得到渲染图
        rendered, grey_rendered, depth, norm, mask = fitter.forward()
        # normalize 深度
        norm_depth = fitter.normalize_depth(depth)
        # 按渲染方式选择loss计算的输入
        if select_input_type == 'grey-rendered':
            loss_input = grey_rendered
        elif select_input_type == 'rendered':
            loss_input = rendered
        elif select_input_type == 'norm':
            loss_input = norm * 0.5 + 0.5

    # 如果是贴图生成阶段
    latent_sds = False
    if fitter.stage == 'texture generation':
        # latent sds阶段
        if iter_step < opt.latent_sds_steps:
            latent_sds = True
        rendered, _, _, _, _ = fitter.forward(render_latent=latent_sds)
        loss_input = rendered        

    if opt.use_view_adjust_prompt and opt.stage != 'edit':
        text_z = text_z[prompt_suffix(fitter.rotation)]
    else:
        text_z = text_z['default']

    # 如果是贴图编辑阶段
    if fitter.stage == 'edit':
        rendered, grey_rendered, depth, norm, mask = fitter.forward(render_latent=False,render_origin_diffuse=True)
        # loss_input = norm * 0.5 + 0.5
        loss_input = rendered
        ip2p_condition_img = fitter.pred_face_origin_diffuse.detach().clone()
        if opt.edit_scope == 'geo':
            # 按渲染方式选择loss计算的输入
            if select_input_type == 'grey-rendered':
                loss_input = grey_rendered
                ip2p_condition_img = fitter.pred_face_origin_grey.detach().clone()
            elif select_input_type == 'rendered':
                loss_input = rendered
                ip2p_condition_img = fitter.pred_face_origin_diffuse.detach().clone()
            elif select_input_type == 'norm':
                loss_input = norm * 0.5 + 0.5
                ip2p_condition_img = fitter.pred_face_origin_normal.detach().clone()


    # 计算损失
    with torch.cuda.amp.autocast(enabled=True):
        t = fitter.scheduler.compute_t(iter_step)

        if fitter.stage == 'coarse geometry generation': 
            loss = fitter.guidance.train_step(text_z, loss_input) # 1, 3, H, W

        if fitter.stage == 'texture generation':
            control_img = None
            if opt.controlnet_name != None:
                mask, depth, _, norm = fitter.render_control_img() # 512x512 control condition
                # normlize depth map
                norm_depth = fitter.normalize_depth_with_camerad(depth).repeat(1,3,1,1)
                # prepare normal map
                bg = torch.zeros(norm.shape).to(norm.device)
                bg[:,2,:,:] = 0.5
                bg[:,0,:,:] = 0.25
                bg[:,1,:,:] = 0.25
                pix_norm = (norm *0.5+0.5) * mask + bg * (1-mask)
                if opt.controlnet_name == 'depth':
                    control_img = norm_depth
                elif opt.controlnet_name == 'normal':
                    control_img = pix_norm
            fitter.control_img = control_img
            loss = fitter.guidance.train_step(text_z, loss_input, guidance_scale=100, set_t=t,input_latent=latent_sds,control_img=control_img) # 1, 3, H, W    

        if fitter.stage == 'edit':
            if opt.attention_sds:
                input_attention_store= attention_store
            else:
                input_attention_store= None
            loss = fitter.guidance.train_step(text_z=text_z,
                                            pred_rgb=loss_input,
                                            condition_img=ip2p_condition_img.detach(),#fitter.pred_face_origin_diffuse.detach(),
                                            prompt_cfg = opt.edit_prompt_cfg, 
                                            image_cfg = opt.edit_img_cfg,
                                            set_t = t,
                                            input_latent=False,
                                            attention_store=input_attention_store,indices_to_alter=indices_to_alter)
            fitter.control_img =None
     
        if employ_textureLDM:
            loss_textureYuvLDM = 0
            if latent_sds:
                texture_sds_input = fitter.diffuse_latent
            else:
                texture_sds_input = fitter.diffuse_texture.permute(0,3,1,2)
                # yuv只有在非latent阶段才会使用
                if opt.employ_yuv:
                    yuv = RgbToYuv()
                    texture_sds_yuv_input = yuv(texture_sds_input)
                    Y_scale = 1
                    # yuv (0~1,-0.5~0.5,-0.5~0.5) -> (-1~1,-1~1,-1~1)
                    texture_sds_yuv_input[:,0] = (texture_sds_yuv_input[:,0]-0.5)*2 * Y_scale
                    texture_sds_yuv_input[:,1] = texture_sds_yuv_input[:,1] * 2
                    texture_sds_yuv_input[:,2] = texture_sds_yuv_input[:,2] * 2

                    text_z_texLDM = static_text_z if opt.use_static_text else text_z
                    loss_textureYuvLDM = fitter.guidance.train_step_textureLDM(text_z_texLDM,texture_sds_yuv_input,guidance_scale=opt.cfg_texSD, set_t = t,input_latent=latent_sds,input_yuv=True)

            text_z_texLDM = static_text_z if opt.use_static_text else text_z
            loss_textureLDM = fitter.guidance.train_step_textureLDM(text_z_texLDM,texture_sds_input,guidance_scale=opt.cfg_texSD, set_t = t,input_latent=latent_sds)

            # regualize loss
            loss_sym = texture_symmetric_loss(texture_sds_input)
            loss_smooth = texture_smooth_loss(texture_sds_input)

            # w_schedule: set the weight of texSD in SDS as a decreasing schedule
            if opt.set_w_schedule:
                if opt.w_schedule == 'linear':
                    # linear
                    interval = (opt.w_texSD_max - opt.w_texSD_min) / total_steps
                    tmp_w_texSD = opt.w_texSD_max - interval * iter_step
                if opt.w_schedule == 'log':
                    # log
                    interval = np.log(opt.w_texSD_max / opt.w_texSD_min) / total_steps
                    tmp_w_texSD = np.exp(np.log(opt.w_texSD_max) - interval * iter_step) 
            else:
                tmp_w_texSD = opt.w_texSD

            loss = loss * opt.w_SD + loss_textureLDM * tmp_w_texSD + loss_textureYuvLDM *opt.w_texYuv + loss_sym * opt.w_sym + loss_smooth * opt.w_smooth


    # edit阶段 正则损失
    if fitter.stage == 'edit':
        # reg_diffuse = torch.nn.functional.mse_loss(fitter.diffuse_texture,fitter.origin_diffuse_texture)

        if iter_step > 40 and opt.attention_reg_diffuse and opt.indices_to_alter != None: # 使用attention aware consistency constraint
            attention_map,_ = attention_mask(fitter.guidance.attentionStore,opt.indices_to_alter,16,512,0.2)
            attention_map = attention_map[0].to('cuda').permute(0,2,3,1)
            # attention_map = attention_map[0].repeat(1,3,1,1)
            UV_attention_mask, UV_attention_map = fitter.bi_direction_projection(fitter.rotation,fitter.translation_z,attention_map,direction='reverse',specific_img_size=512, specific_uv_size=512)
        
            # gs = gaussian_smoothing.GaussianSmoothing(channels=1,kernel_size=3,sigma=0.5,dim=2).to('cuda')


            # attention weight
            UV_attention_mask = UV_attention_mask.permute(0,2,3,1)
            UV_attention_map = UV_attention_map.permute(0,2,3,1)

            if opt.scp_fuse in ['avg2','max2','avm2']:
                UV_attention_weight = 1-UV_attention_map**2  #0.8->0.64,即核心区域一致性会偏高，而非核心的一致性非常高
            elif opt.scp_fuse in ['avg0.5']:
                UV_attention_weight = 1-UV_attention_map**0.5  #0.8->0.9,即核心区域一致性会降低，而非核心一致性会偏低
            else:
                assert opt.scp_fuse in ['avg0.5','avg2','max2','avm2']

            if not hasattr(fitter,'UV_attention_weight'): 
                fitter.UV_attention_weight = torch.ones_like(UV_attention_weight)
                
            fitter.tmp_UV_weight = UV_attention_weight.detach().clone()

            w = 0.8
            if opt.scp_fuse[:3] == 'avg':
            # # uv_weight更新方式： 滑动平均
            # # fix: 更新时，应该只更新当次被投影的区域，其余区域保持不变
                fitter.UV_attention_weight = (fitter.UV_attention_weight * w + UV_attention_weight * (1-w)) * UV_attention_mask + fitter.UV_attention_weight * (1-UV_attention_mask)
            # # fitter.UV_attention_weight = fitter.UV_attention_weight * memtom + UV_attention_weight * (1-memtom)
            elif opt.scp_fuse[:3] == 'max':
                # # uv_weight更新方式： 取所有轮最大值
                update_msk = (fitter.UV_attention_weight > UV_attention_weight).int()
                fitter.UV_attention_weight = fitter.UV_attention_weight * (1-update_msk) + UV_attention_weight * update_msk
            elif opt.scp_fuse[:3] == 'avm':
                # # uv_weight更新方式： 仅在每轮与当前结果的最大值上使用滑动平均
                update_msk = (fitter.UV_attention_weight > UV_attention_weight).int()
                fitter.UV_attention_weight = fitter.UV_attention_weight * (1-update_msk) + \
                                            (UV_attention_weight * (1-w) + fitter.UV_attention_weight * w) * update_msk
                


        else:
            fitter.UV_attention_weight = torch.ones_like(fitter.diffuse_texture)        

        if opt.edit_scope == 'tex':
            reg_consistency = (fitter.UV_attention_weight * (fitter.diffuse_texture - fitter.origin_diffuse_texture) ** 2).mean()
        elif opt.edit_scope == 'geo':
            if not hasattr(fitter,'origin_pos_map'):
                fitter.origin_pos_map = fitter.shape2posmap(fitter.load_vertex[0])
            fitter.update_pos_map = fitter.shape2posmap(fitter.pred_vertex_no_pose[0])
            reg_consistency = (fitter.UV_attention_weight * (fitter.update_pos_map - fitter.origin_pos_map) ** 2).mean()

        # # time-step decreasing weight
        # interval = (1 - 0.3) / total_steps
        # tmp_w_reg = 1 - interval * iter_step
        # loss+= opt.w_reg_diffuse * reg_diffuse * tmp_w_reg
        loss += opt.w_reg_diffuse * reg_consistency

    return loss


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def main():
    opt = parse_args()
    seed_num = opt.seed
    seed_everything(seed_num)
    print(f'Welcome to text2face !!! random seed:{seed_num}')
    # exp setttings
    device = opt.device
    total_steps = opt.total_steps
    save_freq = opt.save_freq
    exp_root = opt.exp_root
    exp_name = opt.exp_name

    scaler = torch.cuda.amp.GradScaler(enabled=True) # use mixed precision training 

    employ_textureLDM = False
    if opt.textureLDM_path and opt.stage == 'texture generation':
        employ_textureLDM = True
    if opt.textureLDM_path and opt.stage == 'edit' and opt.edit_scope == 'tex':
        employ_textureLDM = True

    if opt.stage == 'edit':
        employ_instructp2p = True
    else:
        employ_instructp2p = False

    if opt.guidance_type == 'clip':
        guidance = CLIP(device)
    elif opt.guidance_type == 'stable-diffusion':
        if employ_instructp2p:
            guidance = StableDiffusion_instructp2p(device,True,False,textureLDM_path=opt.textureLDM_path,textureLDM_yuv_path=opt.textureLDM_yuv_path)
        elif employ_textureLDM:
            guidance = StableDiffusion_w_texture(device, True, False, sd_version=opt.sd_version,
                                                textureLDM_path=opt.textureLDM_path,controlnet_name=opt.controlnet_name,textureLDM_yuv_path=opt.textureLDM_yuv_path)
        else:
            guidance = StableDiffusion(device, True, False, sd_version=opt.sd_version)  # use float32 for training  # fp16 vram_optim
        

    if opt.vis_att:
        import lib.boxdiff.ptp_utils as ptp_utils
        import lib.boxdiff.vis_utils as vis_utils
        from lib.boxdiff.ptp_utils import AttentionStore
        controller = AttentionStore()
        ptp_utils.register_attention_control(guidance, controller)
        guidance.attentionStore = controller
    else:
        guidance.attentionStore = None

    # user input indices to alter only used in edit stage
    if opt.stage == 'edit':
        opt.indices_to_alter = util.get_indices_to_alter(guidance,opt.text,opt.indices_to_alter_str)
    else:
        opt.indices_to_alter = None

    # StageFitter for different stages of the pipeline
    fitter = StageFitter(SD_guidance = guidance,
                            stage=opt.stage,diffuse_generation_type=opt.texture_generation,
                            render_resolution=opt.render_resolution,
                            saved_id_path=opt.load_id_path,saved_dp_path=opt.load_dp_path,saved_diffuse_path=opt.load_diffuse_path,
                            latent_init=opt.latent_init,dp_map_scale=opt.dp_map_scale,edit_scope=opt.edit_scope)    

    # using SDS -- a normal decreasing schedule in denoise process
    ts = T_scheduler(opt.schedule_type,total_steps,max_t_step = guidance.scheduler.config.num_train_timesteps)

    fitter.scheduler = ts

    fitter.employ_textureLDM = employ_textureLDM
    fitter.employ_instructp2p = employ_instructp2p
    # viewpoint settings
    fitter.set_transformation_range(x_min_max=[opt.viewpoint_range_X_min,opt.viewpoint_range_X_max],
                                    y_min_max=[opt.viewpoint_range_Y_min,opt.viewpoint_range_Y_max],
                                    z_min_max=[opt.viewpoint_range_Z_min,opt.viewpoint_range_Z_max],
                                    t_z_min_max=[opt.t_z_min,opt.t_z_max])


    fitter.random_view_with_choice = False # for geo
    if opt.force_fixed_viewpoint:
        if opt.stage == 'texture generation':
            fitter.set_transformation_choices(x_list=[0,-30],y_list=[0,60,120,240,300],z_list=[0],t_z_list=[1.5,3])
            fitter.random_view_with_choice = True # for tex
        if opt.stage == 'edit':
            fitter.set_transformation_choices(x_list=[0,-10],y_list=[0,30,330,60,300],z_list=[0],t_z_list=[1.5,3])
            if opt.edit_scope == 'geo':
                fitter.set_transformation_choices(x_list=[0,-10,-20],y_list=[0,60,300,30,330],z_list=[0],t_z_list=[1.5])   
            fitter.random_view_with_choice = True # for edit


    fitter.to(device)

    # prompt settings
    text = opt.text
    negative_text = opt.negative_text

    # sds loss rendering settings
    sds_input = opt.sds_input

    # save folder setting
    exp_folder = os.path.join(exp_root,exp_name,text,opt.stage,f'seed{seed_num}')

    # set exp_name to '' for next usage
    exp_name = ''
    if opt.stage == 'coarse geometry generation':
        if opt.path_debug:
            exp_name += f'input_{sds_input}'

    if opt.stage == 'texture generation':
        exp_folder = os.path.join(exp_folder,opt.texture_generation)

        if opt.path_debug:
            if opt.set_w_schedule:  #dynamic w_texSD 
                exp_name = f'Wschedule_{opt.w_schedule}_max{opt.w_texSD_max}_min{opt.w_texSD_min}'
            else:                   #fixed w_texSD
                exp_name = f'w_texSD{opt.w_texSD}'
            
            exp_name += f'_sym{opt.w_sym}_smooth{opt.w_smooth}'

            exp_name += f'_cfg_texSD{opt.cfg_texSD}'

            if not opt.use_static_text:
                exp_name += "_supervisedTex"

            if opt.set_t_schedule:  #schedule-dynamic timestep instead random timestep
                exp_name += '_Tschedule'
            # exp_name += opt.schedule_type
            
            if opt.latent_sds_steps > 0:
                exp_name += f'_la{opt.latent_sds_steps}'

            if opt.controlnet_name:
                exp_name += f'_Cont0{opt.controlnet_name}'

            if opt.use_view_adjust_prompt:
                exp_name += '_VDPrompt'
            if opt.employ_yuv:
                exp_name += f'_w_texYuv{opt.w_texYuv}'

    if opt.stage == 'edit':
        exp_name += opt.edit_scope
        if opt.path_debug:
            exp_name += f'_promptcfg{opt.edit_prompt_cfg}'
            exp_name += f'_imgcfg{opt.edit_img_cfg}'
            exp_name += f'_diffuseReg{opt.w_reg_diffuse}'
            if opt.set_w_schedule:  #dynamic w_texSD 
                exp_name += f'Wschedule_{opt.w_schedule}_max{opt.w_texSD_max}_min{opt.w_texSD_min}'
            else:                   #fixed w_texSD
                exp_name += f'w_texSD{opt.w_texSD}'

            if opt.employ_yuv:
                exp_name += f'_w_texYuv{opt.w_texYuv}'
            if opt.attention_reg_diffuse:
                exp_name += '_attreg'
            if opt.attention_sds:
                exp_name += '_attsds'
            
            exp_name += opt.scp_fuse

        # exp_folder = pathlib.Path(opt.load_diffuse_path).parent
        # exp_folder = os.path.join(exp_folder,text)
    exp_folder = os.path.join(exp_folder,exp_name)

    os.makedirs(exp_folder, exist_ok=True)
    os.makedirs(os.path.join(exp_folder,'display'), exist_ok=True)

    # get guidance text embedding
    if opt.stage == 'edit':
        text_z = {'default':guidance.get_text_embeds_for_instructp2p(text, negative_text)}
        static_text_z = guidance.get_text_embeds(opt.static_text, negative_text)
    else:
        text_z = {
            'default':guidance.get_text_embeds(text, negative_text),
            'front view': guidance.get_text_embeds(text+', front view', negative_text),
            'back view': guidance.get_text_embeds(text+', back view', negative_text),
            'side view': guidance.get_text_embeds(text+', side view', negative_text)
        }
        static_text_z = guidance.get_text_embeds(opt.static_text, negative_text)


    lr = opt.lr    
    # save training info
    with open(os.path.join(exp_folder, 'training.json'), 'w') as file:
        json.dump(opt.__dict__,file, indent=2)

    if opt.stage in['texture generation','edit']:
        from torch.optim import AdamW
        optim = torch.optim.AdamW(fitter.get_parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-15)
        scheduler = StepLR(optim, step_size=100, gamma=0.9)
    else:
        from lib.optimizer import Adan
        # Adan usually requires a larger LR
        # optim = Adan(fitter.get_parameters_pose_fixed(), lr=lr, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        optim = Adan(fitter.get_parameters(), lr=lr, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)    
        scheduler = StepLR(optim, step_size=100, gamma=1.0)


    for iter_step in tqdm(range(total_steps)):
        optim.zero_grad()
        loss = train_step(fitter,text , text_z,static_text_z, opt, sds_input=sds_input,employ_textureLDM=fitter.employ_textureLDM, iter_step=iter_step, total_steps=total_steps,
                            attention_store=guidance.attentionStore,indices_to_alter=opt.indices_to_alter)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scheduler.step()
        scaler.update()


        if (iter_step) % save_freq == 0:
            with torch.no_grad():
                fitter.forward(random_sample_view=False) # Keep the same perspective as the previous iter so that subsequent vis attention is aligned

                fitter.save_attention(exp_folder,iter_step,text)

                fitter.save_visuals(exp_folder,iter_step,
                                rx=opt.display_rotation_x,ry=opt.display_rotation_y,rz=opt.display_rotation_z,tz=opt.display_translation_z)
                                  
                fixed_shape = (opt.stage == 'texture generation')
                if not fixed_shape or iter_step >= total_steps-save_freq:
                    fitter.save_results(exp_folder,iter_step,save_mesh=True,save_npy=True)

if __name__ == '__main__':
    
    main()
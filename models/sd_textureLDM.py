from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline,ControlNetModel
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
# from util.util import timing
# suppress partial model loading warning
logging.set_verbosity_error()

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd
from dataclasses import dataclass


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

@dataclass
class UNet2DConditionOutput:
    sample: torch.HalfTensor # Not sure how to check what unet_traced.pt contains, and user wants. HalfTensor or FloatTensor

class StableDiffusion_w_texture(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, textureLDM_path=None, textureLDM_yuv_path=None, controlnet_name=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')

        # if hf_key is not None:
        #     print(f'[INFO] using hugging face custom model key: {hf_key}')
        #     model_key = hf_key
        # elif self.sd_version == '2.1':
        #     model_key = "stabilityai/stable-diffusion-2-1-base"
        # elif self.sd_version == '2.0':
        #     model_key = "stabilityai/stable-diffusion-2-base"
        # elif self.sd_version == '1.5':
        #     model_key = "runwayml/stable-diffusion-v1-5"
        # else:
        #     raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # textureLDM use v1-4 trained
        # model_key = 'CompVis/stable-diffusion-v1-4'
        model_key = "runwayml/stable-diffusion-v1-5"

        precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=precision_t,safety_checker=None)
        pipe.unet_texture = UNet2DConditionModel.from_pretrained(textureLDM_path, torch_dtype=precision_t)

        if textureLDM_yuv_path:
            pipe.unet_yuv_texture = UNet2DConditionModel.from_pretrained(textureLDM_yuv_path, torch_dtype=precision_t)


        self.use_controlnet=False
        if controlnet_name == 'normal':
            self.controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-normal',torch_dtype=precision_t)
            self.use_controlnet=True
            self.controlnet.to(device)
        if controlnet_name == 'depth':
            self.controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-depth',torch_dtype=precision_t)
            self.use_controlnet=True
            self.controlnet.to(device)
        if controlnet_name == 'pose':
            self.controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-openpose',torch_dtype=precision_t)
            self.use_controlnet=True
            self.controlnet.to(device)
        if controlnet_name == 'densepose':
            self.controlnet = ControlNetModel.from_single_file("/project/mengyapeng/controlnetFor_v10.safetensors")  # 加dtype报错
            self.use_controlnet=True
            self.controlnet.to(device)


        if isfile('./unet_traced.pt'):
            # use jitted unet
            unet_traced = torch.jit.load('./unet_traced.pt')
            class TracedUNet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.in_channels = pipe.unet.in_channels
                    self.device = pipe.unet.device

                def forward(self, latent_model_input, t, encoder_hidden_states):
                    sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
                    return UNet2DConditionOutput(sample=sample)
            pipe.unet = TracedUNet()

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            if is_xformers_available():
                pipe.enable_xformers_memory_efficient_attention()
            pipe.to(device)
            pipe.unet_texture.to(device)

            if textureLDM_yuv_path:
                pipe.unet_yuv_texture.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.unet_texture = pipe.unet_texture
        if textureLDM_yuv_path:
            self.unet_yuv_texture = pipe.unet_yuv_texture


        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=precision_t)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings


    def train_step(self, text_embeddings, pred_rgb, guidance_scale = 100,set_t=None,input_latent=False,control_img=None): #_regular_unet=100, guidance_scale_texture_unet=100):
        
        # interp to 512x512 to be fed into vae.
        torch.cuda.synchronize()
        tmp = time.time()


        # latent sds
        if input_latent:
            latents = pred_rgb
        # image sds
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if set_t:
            # assert set_t >= self.min_step and set_t <= self.max_step
            t = torch.tensor([set_t],dtype=torch.long,device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        # t = torch.randint(t_min,self.max_step+1,[1],dtype=torch.long,device=self.device)
        
        # print('3time:%3f'%(time.time()-now))
        # now = time.time()
        # encode image into latents with vae, requires grad!
        # tmp = timing('encode img',tmp)
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # tmp = timing('add noise',tmp)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            
            #----controlnet
            if self.use_controlnet:
                assert control_img != None
                control_model_input = latent_model_input
                controlnet_prompt_embeds = text_embeddings
                with torch.cuda.amp.autocast(enabled=True):
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=control_img,
                        conditioning_scale=1,
                        guess_mode=False,
                        return_dict=False,
                    )
                    noise_pred = self.unet(
                        latent_model_input,t,encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]
            # Save input tensors for UNet
            #torch.save(latent_model_input, "train_latent_model_input.pt")
            #torch.save(t, "train_t.pt")
            #torch.save(text_embeddings, "train_text_embeddings.pt")
            else:
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # noise_pred_texture = self.unet_texture(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # tmp = timing('unet pred noise',tmp)
        # print('4time:%3f'%(time.time()-now))
        # now = time.time()
        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # noise_pred_uncond_texture, noise_pred_text_texture = noise_pred_texture.chunk(2)
        # noise_pred_texture = noise_pred_text_texture + guidance_scale_texture_unet * (noise_pred_text_texture - noise_pred_uncond_texture)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)
        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)
        # tmp = timing('compute loss',tmp)
        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)
        # tmp = timing('apply loss',tmp)
        # print('9time:%3f'%(time.time()-now))
        # grad_texture = w*(noise_pred_texture - noise)
        # grad_texture = torch.nan_to_num(grad_texture)
        # loss_texture = SpecifyGradient.apply(latents,grad_texture)
        return loss#, loss_texture



    def train_step_textureLDM(self, text_embeddings, pred_rgb, guidance_scale=100,set_t=None,input_latent=False,input_yuv=False):
        
        torch.cuda.synchronize()
        tmp = time.time()

        # latent sds
        if input_latent:
            latents = pred_rgb
        # image sds
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # tmp = timing('interpolate',tmp)
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if set_t:
            # assert set_t >= self.min_step and set_t <= self.max_step
            t = torch.tensor([set_t],dtype=torch.long,device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        # t = torch.randint(t_min,self.max_step+1,[1],dtype=torch.long,device=self.device)

        # print('3time:%3f'%(time.time()-now))
        # now = time.time()
        # tmp = timing('encode img',tmp)
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # tmp = timing('add noise',tmp)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # Save input tensors for UNet
            #torch.save(latent_model_input, "train_latent_model_input.pt")
            #torch.save(t, "train_t.pt")
            #torch.save(text_embeddings, "train_text_embeddings.pt")
            # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            if input_yuv:
                noise_pred = self.unet_yuv_texture(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            else:              
                noise_pred = self.unet_texture(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # tmp = timing('unet pred noise',tmp)
        # print('4time:%3f'%(time.time()-now))
        # now = time.time()
        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)
        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)
        # tmp = timing('compute loss',tmp)
        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)
        # tmp = timing('apply loss',tmp)
        # print('9time:%3f'%(time.time()-now))
        return loss

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None, textureLDM=False):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # Save input tensors for UNet
                #torch.save(latent_model_input, "produce_latents_latent_model_input.pt")
                #torch.save(t, "produce_latents_t.pt")
                #torch.save(text_embeddings, "produce_latents_text_embeddings.pt")
                # predict the noise residual
                with torch.no_grad():
                    if textureLDM:
                        noise_pred = self.unet_texture(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
                    else:
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None, textureLDM=False):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, textureLDM=textureLDM) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs

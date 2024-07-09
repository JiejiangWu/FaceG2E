import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline,StableDiffusionPipeline
from typing import Callable, List, Optional, Union
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.utils import (
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
)
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def download_image(url):
   image = PIL.Image.open(requests.get(url, stream=True).raw)
   image = PIL.ImageOps.exif_transpose(image)
   image = image.convert("RGB")
   return image

def test_ip2p():
    model_id = "timbrooks/instruct-pix2pix"  # <- replace this
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    generator = torch.Generator("cuda").manual_seed(0)

    url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
    image = download_image(url)
    image.save("origin.png")
    prompt = "turn him into cyborg"
    num_inference_steps = 20
    image_guidance_scale = 1.5
    guidance_scale = 10

    edited_image = pipe(
    prompt,
    image=image,
    num_inference_steps=num_inference_steps,
    image_guidance_scale=image_guidance_scale,
    guidance_scale=guidance_scale,
    generator=generator,
    ).images[0]
    edited_image.save("edited_image.png")

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


@dataclass
class UNet2DConditionOutput:
    sample: torch.HalfTensor # Not sure how to check what unet_traced.pt contains, and user wants. HalfTensor or FloatTensor

class StableDiffusion_instructp2p(nn.Module):
    def __init__(self, device, fp16, vram_O, model_key='timbrooks/instruct-pix2pix', hf_key=None, textureLDM_path=None, textureLDM_yuv_path=None, controlnet_name = None):
        super().__init__()

        self.device = device

        print(f'[INFO] loading instruct_pix2pix...')

        precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_key, torch_dtype=precision_t, use_safetensors=True,safety_checker=None).to("cuda")

        if textureLDM_path:
            pipe.unet_texture = UNet2DConditionModel.from_pretrained(textureLDM_path, torch_dtype=precision_t)
        if textureLDM_yuv_path:
            pipe.unet_yuv_texture = UNet2DConditionModel.from_pretrained(textureLDM_yuv_path, torch_dtype=precision_t)


        pipe_sd = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=precision_t,safety_checker=None)
        self.use_controlnet=False
        if controlnet_name == 'normal':
            self.controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-normal',torch_dtype=precision_t)
            self.use_controlnet=True
            self.controlnet.to(device)
        if controlnet_name == 'depth':
            self.controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-depth',torch_dtype=precision_t)
            self.use_controlnet=True
            self.controlnet.to(device)
        


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
            pipe_sd.unet.to(device)
            if textureLDM_path:
                pipe.unet_texture.to(device)
            if textureLDM_yuv_path:
                pipe.unet_yuv_texture.to(device)
        
        self.vae = pipe.vae
        # self.vae = pipe_sd.vae #测试vae是否在sd和instructp2p之间通用
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.unet_sd = pipe_sd.unet
        
        if textureLDM_path:
            self.unet_texture = pipe.unet_texture
        if textureLDM_yuv_path:
            self.unet_yuv_texture = pipe.unet_yuv_texture

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=precision_t)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds_for_instructp2p(self, prompt, negative_prompt):
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
        text_embeddings = torch.cat([text_embeddings,uncond_embeddings, uncond_embeddings])
        return text_embeddings

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

    def train_step_sd(self, text_embeddings, pred_rgb, guidance_scale = 100,set_t=None,input_latent=False,control_img=None): #_regular_unet=100, guidance_scale_texture_unet=100):
        
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
                    noise_pred = self.unet_sd(
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
                noise_pred = self.unet_sd(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
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

    def produce_latents(self,pred_rgb):
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512)
        return latents

    def train_step(self, text_z, pred_rgb, condition_img, prompt_cfg = 100, image_cfg=2, set_t=None,input_latent=False,
                    attention_store=None,indices_to_alter=None): #_regular_unet=100, guidance_scale_texture_unet=100):
        
        '''
        pred_rgb是在迭代中更新的img，用它来生成latents
        condition_img是最开始的img，用它来生成condition latent
        '''

        # interp to 512x512 to be fed into vae.
        torch.cuda.synchronize()
        tmp = time.time()

        # Prompts -> text embeds
        # text_embeds = self.get_text_embeds_for_instructp2p(prompts, negative_prompts) # [3, 77, 768]   text, uncond_text, uncond_text
        text_embeds = text_z

        # image condition latent
        condition_img_512 = F.interpolate(condition_img, (512, 512), mode='bilinear', align_corners=False)
        condition_img_latents = self.prepare_image_latents(condition_img_512 * 2-1) # 3,4,64,64  img,img,uncondition_img

        # latent sds
        if input_latent:
            img_latents = pred_rgb
        # image sds
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # img_latents = condition_img_latents[0:1,:,:,:].clone()
            img_latents = self.encode_imgs(pred_rgb_512)       # 1,4,64,64  img
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if set_t:
            # assert set_t >= self.min_step and set_t <= self.max_step
            t = torch.tensor([set_t],dtype=torch.long,device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        # t = torch.randint(t_min,self.max_step+1,[1],dtype=torch.long,device=self.device)

        with torch.no_grad():
            # add noise
            noise = torch.randn_like(img_latents)
            latents_noisy = self.scheduler.add_noise(img_latents, noise, t) # 1,4,64,64
            # tmp = timing('add noise',tmp)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 3)             # 3,4,64,64
            
            # concat latents, image_latents in the channel dimension
            # scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            scaled_latent_model_input = torch.cat([latent_model_input, condition_img_latents], dim=1)

            noise_pred = self.unet(
                scaled_latent_model_input, t, encoder_hidden_states=text_embeds, return_dict=False
            )[0]
            
            # noise_pred_texture = self.unet_texture(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # tmp = timing('unet pred noise',tmp)
        # print('4time:%3f'%(time.time()-now))
        # now = time.time()
        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
        noise_pred = (
                noise_pred_uncond
                + prompt_cfg * (noise_pred_text - noise_pred_image)
                + image_cfg * (noise_pred_image - noise_pred_uncond)
            )
        # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        # noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # noise_pred_uncond_texture, noise_pred_text_texture = noise_pred_texture.chunk(2)
        # noise_pred_texture = noise_pred_text_texture + guidance_scale_texture_unet * (noise_pred_text_texture - noise_pred_uncond_texture)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)
        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)

        # multiply attention_weight if use it
        if attention_store != None and indices_to_alter != None:
            from util.loss import attention_mask
            attention_map,_ = attention_mask(attention_store,indices_to_alter,16,64,0.2)
            attention_map = attention_map[0].repeat(1,4,1,1).to(grad.device).detach()
        else:
            attention_map = torch.ones_like(grad)
        grad = grad*attention_map


        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(img_latents, grad)
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

    def img_to_img(self, images, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5,image_guidance_scale=2, latents=None, textureLDM=False):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds_for_instructp2p(prompts, negative_prompts) # [3, 77, 768]

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device='cuda')
        timesteps = self.scheduler.timesteps

        # 5. Images -> img latents
        images = images * 2-1
        image_latents = self.prepare_image_latents(images)

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = torch.randn((text_embeds.shape[0] // 2, num_channels_latents, height // 8, width // 8), device=self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # 7. Check that shapes of latents and image match the UNet channels
        num_channels_image = image_latents.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )
        # 9. Denoising loop
        timesteps = self.scheduler.timesteps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 3)

            # concat latents, image_latents in the channel dimension
            scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

            # predict the noise residual
            noise_pred = self.unet(
                scaled_latent_model_input, t, encoder_hidden_states=text_embeds, return_dict=False
            )[0]

            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)[0]


         # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


    def prepare_image_latents(
        self, image, batch_size=1, num_images_per_prompt=1, device='cuda', do_classifier_free_guidance=True, generator=None
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            image_latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                image_latents = [self.vae.encode(image[i : i + 1]).latent_dist.mode() for i in range(batch_size)]
                image_latents = torch.cat(image_latents, dim=0)
            else:
                image_latents = self.vae.encode(image).latent_dist.mode()

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand image_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            print(deprecation_message)
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        if do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents




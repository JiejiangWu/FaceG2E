from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline,ControlNetModel,StableDiffusionControlNetPipeline
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from models.sd import StableDiffusion
from util.util import timing
# suppress partial model loading warning
logging.set_verbosity_error()
import torchvision.transforms as transforms

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd
from dataclasses import dataclass

class ControlNet(nn.Module):
    def __init__(self, device, model_name='normal'):
        super().__init__()
        if model_name == 'normal':
            self.controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-normal')
        if model_name == 'depth':
            self.controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-depth')
        if model_name == 'pose':
            self.controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-openpose')


        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
             "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        
        self.device = device
        self.pipe.to(self.device)
        self.controlnet.to(self.device)
        self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config)

        self.transform = transforms.Compose([
             transforms.PILToTensor()
            ])

    def prompt_to_img(self,prompt,condition_img):
        image = self.pipe(prompt, condition_img, num_inference_steps=50).images[0]
        return self.transform(image)
    
    
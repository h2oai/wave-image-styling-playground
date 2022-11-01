import gc
from io import BytesIO
from typing import Optional

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image
from torch import autocast


def generate_image_with_prompt(input_img_path: Optional[str]=None, prompt_txt: str = "Face portrait",
                                                        output_path: str=None):
    # License: https://huggingface.co/spaces/CompVis/stable-diffusion-license
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "./models/stable_diffusion_v1_4"

    if input_img_path:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, revision="fp16",
                                                                    torch_dtype=torch.float16).to(device)
        # Open image
        image_input = Image.open(input_img_path).convert("RGB")
        init_image = image_input.resize((512, 512))

        with autocast(device):
            images = pipe(prompt=prompt_txt, init_image=init_image, strength=0.5, guidance_scale=7.5)["sample"]
    else: # Default prompt
        pipe = StableDiffusionPipeline.from_pretrained(model_path, revision="fp16",
                                                                    torch_dtype=torch.float16).to(device)

        with autocast(device):
            images = pipe(prompt=prompt_txt).images


    file_name = output_path + '/result.jpg'
    if output_path:
        images[0].save(file_name)
    gc.collect()
    torch.cuda.empty_cache()
    return file_name

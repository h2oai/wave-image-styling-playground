import gc
import torch
from torch import autocast
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline


def generate_image_with_prompt(input_img_path: str, prompt_txt: str = "Face portrait",
                                                        output_path: str=None, save: bool=True):
    torch.cuda.empty_cache()
    device = "cuda"
    # License: https://huggingface.co/spaces/CompVis/stable-diffusion-license
    model_path = "./models/stable_diffusion_v1_4"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, revision="fp16",
                                                                torch_dtype=torch.float16,)
    pipe = pipe.to(device)

    # Open image
    image_input = Image.open(input_img_path).convert("RGB")
    init_image = image_input.resize((512, 512))

    with autocast(device):
        images = pipe(prompt=prompt_txt, init_image=init_image, strength=0.5, guidance_scale=7.5)["sample"]

    file_name = output_path + '/result.jpg'
    images[0].save(file_name)
    gc.collect()
    torch.cuda.empty_cache()
    return file_name

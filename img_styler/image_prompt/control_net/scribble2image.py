# https://github.com/lllyasviel/ControlNet/blob/main/gradio_scribble2image.py

import gc
import os

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from img_styler.image_prompt.control_net.annotator.util import resize_image, HWC3
from img_styler.image_prompt.control_net.cldm.model import create_model, load_state_dict
from img_styler.image_prompt.control_net.cldm.ddim_hacked import DDIMSampler


def get_scribble_image_samples(
    input_img_path,
    prompt,
    seed,
    output_path="",
    a_prompt="",
    n_prompt="",
    num_samples=1,
    image_resolution=256,
    ddim_steps=20,
    guess_mode=False,
    strength=1.0,
    scale=5.0,
    eta=0.0,
    save_memory=True,
):
    input_image = cv2.imread(input_img_path)

    dirname = os.path.dirname(__file__)
    model = create_model(os.path.join(dirname, "models/cldm_v15.yaml")).cpu()
    model.load_state_dict(load_state_dict("models/controlnet/control_sd15_scribble.pth", location="cuda"))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, _ = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
        )

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        results = [x_samples[i] for i in range(num_samples)]

        gc.collect()

        if output_path:
            filename = os.path.join(output_path, "result.jpg")
            cv2.imwrite(filename, results[0])
            return filename

    return [255 - detected_map] + results

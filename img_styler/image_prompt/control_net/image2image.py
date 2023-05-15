# Reference: https://github.com/lllyasviel/ControlNet/blob/main/gradio_canny2image.py
import gc
import os

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from img_styler.image_prompt.control_net.annotator.hed import HEDdetector, nms
from img_styler.image_prompt.control_net.annotator.midas import MidasDetector
from img_styler.image_prompt.control_net.annotator.mlsd import MLSDdetector
from img_styler.image_prompt.control_net.annotator.openpose import OpenposeDetector
from img_styler.image_prompt.control_net.annotator.uniformer import UniformerDetector
from img_styler.image_prompt.control_net.annotator.util import resize_image, HWC3
from img_styler.image_prompt.control_net.annotator.canny import CannyDetector
from img_styler.image_prompt.control_net.cldm.model import create_model, load_state_dict
from img_styler.image_prompt.control_net.cldm.ddim_hacked import DDIMSampler


class ControlNetMode:
    CANNY = "canny"
    SCRIBBLE = "scribble"
    DEPTH = "depth"
    HED = "hed"
    HOUGH = "hough"
    NORMAL = "normal"
    POSE = "pose"
    SEG = "seg"
    FAKE_SCRIBBLE = "fake_scribble"


def get_controlnet_image_samples(
    input_img_path,
    prompt,
    seed,
    mode=ControlNetMode.CANNY,
    output_path="",
    a_prompt="",
    n_prompt="",
    num_samples=1,
    image_resolution=256,
    detect_resolution=256,
    ddim_steps=20,
    guess_mode=False,
    strength=1.0,
    scale=5.0,
    eta=0.0,
    low_threshold=255 / 3,
    high_threshold=255,
    value_threshold=0.1,
    distance_threshold=0.1,
    bg_threshold=0.4,
    save_memory=True,
):
    input_image = cv2.imread(input_img_path)

    dirname = os.path.dirname(__file__)
    model = create_model(os.path.join(dirname, "models/cldm_v15.yaml")).cpu()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, _ = img.shape

        if mode == ControlNetMode.CANNY:
            model_path = "models/controlnet/control_sd15_canny.pth"
            apply_canny = CannyDetector()
            detected_map = apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)
        elif mode == ControlNetMode.DEPTH:
            model_path = "models/controlnet/control_sd15_depth.pth"
            apply_midas = MidasDetector()
            detected_map, _ = apply_midas(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        elif mode == ControlNetMode.SCRIBBLE:
            model_path = "models/controlnet/control_sd15_scribble.pth"
            detected_map = np.zeros_like(img, dtype=np.uint8)
            detected_map[np.min(img, axis=2) < 127] = 255
        elif mode == ControlNetMode.HED:
            model_path = "models/controlnet/control_sd15_hed.pth"
            apply_hed = HEDdetector()
            detected_map = apply_hed(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        elif mode == ControlNetMode.HOUGH:
            model_path = "models/controlnet/control_sd15_mlsd.pth"
            apply_mlsd = MLSDdetector()
            detected_map = apply_mlsd(resize_image(input_image, detect_resolution), value_threshold, distance_threshold)
            detected_map = HWC3(detected_map)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        elif mode == ControlNetMode.NORMAL:
            model_path = "models/controlnet/control_sd15_normal.pth"
            apply_midas = MidasDetector()
            _, detected_map = apply_midas(resize_image(input_image, detect_resolution), bg_th=bg_threshold)
            detected_map = HWC3(detected_map)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        elif mode == ControlNetMode.POSE:
            model_path = "models/controlnet/control_sd15_openpose.pth"
            apply_openpose = OpenposeDetector()
            detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        elif mode == ControlNetMode.SEG:
            model_path = "models/controlnet/control_sd15_seg.pth"
            apply_uniformer = UniformerDetector()
            detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        elif mode == ControlNetMode.FAKE_SCRIBBLE:
            model_path = "models/controlnet/control_sd15_scribble.pth"
            apply_hed = HEDdetector()
            detected_map = apply_hed(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
            detected_map = nms(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0

        model.load_state_dict(load_state_dict(model_path, location=device))
        model = model.to(device)
        ddim_sampler = DDIMSampler(model)

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
            model.low_vram_shift(is_diffusing=True)

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

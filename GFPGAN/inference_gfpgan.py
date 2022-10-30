from typing import Union
import cv2
from loguru import logger
import numpy as np
import os
import torch
from basicsr.utils import imwrite

from GFPGAN.gfpgan import GFPGANer


def init_gfpgan():
    """Initialize GFPGAN."""
    # ------------------------ set up background upsampler ------------------------
    if not torch.cuda.is_available():  # CPU
        import warnings
        warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                        'If you really want to use it, please modify the corresponding codes.')
        bg_upsampler = None
    else:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True)  # need to set False in CPU mode

    # ------------------------ set up GFPGAN restorer ------------------------
    arch = 'clean'
    channel_multiplier = 2
    model_name = 'GFPGANv1.4'
    url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'

    # determine model paths
    model_path = os.path.join('models/gfpgan', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url
        
    restorer = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)
    
    return restorer


def restore_image(restorer, input_img: Union[np.ndarray, str], output_img_path: str, weight: float = 0.5, has_aligned: bool = False):
    if isinstance(input_img, str):
        input_img = cv2.imread(input_img)
    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=has_aligned,
        only_center_face=True,
        paste_back=True,
        weight=weight)

    # save faces
    suffix = 'restored'
    for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
        basename, _ = os.path.splitext(output_img_path)
        save_face_name = f'{basename}_{suffix}_{idx:02d}.png'
        imwrite(restored_face, save_face_name)

    # save restored img
    if restored_img is not None:
        imwrite(restored_img, output_img_path)
        logger.debug(f'Results saved to {output_img_path}.')

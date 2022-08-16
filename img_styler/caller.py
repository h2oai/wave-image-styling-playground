# Reference: https://github.com/renatoviolin/webapp-StyleGAN2-ADA-PyTorch
import copy
import logging
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torchvision import utils
from PIL import Image
from loguru import logger
from pathlib import Path

from .dnnlib.util import open_url
from .face_aligner import align_face
from .sg2_model import Generator

sys.path.append(f"{os.getcwd()}/img_styler")
device = torch.device('cuda')

logger.info('Load pre-trained model...')
with open(f'./models/ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)
OUTPUT_PATH = "./var/lib/tmp/jobs/output"


def apply_projection(proj_a, proj_b, idx_to_swap=(0, 3)):
    ws_a = np.load(proj_a)['w']
    ws_b = np.load(proj_b)['w']
    ws_a[0, idx_to_swap, :] = ws_b[0, idx_to_swap, :]
    return ws_a


def synthesize_new_img(projection):
    if type(projection).__module__ != np.__name__:
        ws = np.load(projection)['w']
    else:
        ws = projection

    ws = torch.tensor(ws, device=device).squeeze(0)
    img = G.synthesis(ws.unsqueeze(0), noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    img = Image.fromarray(img[0].cpu().numpy(), 'RGB')
    return img


# Reference: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/projector.py
def __project(
    G,
    target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps=1000,
    w_avg_samples=10000,
    initial_learning_rate=0.1,
    initial_noise_factor=0.05,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    regularize_noise_weight=1e5,
    verbose=False,
    seed: int = 42,
    device: torch.device,
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)  # type: ignore

    # Compute w stats.
    logger.info(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(seed).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = {
        name: buf
        for (name, buf) in G.synthesis.named_buffers()
        if 'noise_const' in name
    }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(
        w_avg, dtype=torch.float32, device=device, requires_grad=True
    )  # pylint: disable=not-callable
    w_out = torch.zeros(
        [num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device
    )
    optimizer = torch.optim.Adam(
        [w_opt] + list(noise_bufs.values()),
        betas=(0.9, 0.999),
        lr=initial_learning_rate,
    )

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = (
            w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        )
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logger.info(
            f'Step {step + 1:>4d}/{num_steps}: Dist {dist:<4.2f} Loss {float(loss):<5.2f}'
        )

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])


def generate_projection(input_img, outdir: str, n_steps=1000, random_state: int = 42):
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Load input image.
    target_pil = align_face(input_img)
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(
        ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
    )
    logger.info(f"Adjusting img resolution: {G.img_resolution}*{G.img_resolution}")
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    projected_w_steps = __project(
        G,
        target=torch.tensor(
            target_uint8.transpose([2, 0, 1]), device=device
        ),  # pylint: disable=not-callable
        num_steps=n_steps,
        initial_learning_rate=0.01,
        device=device,
        verbose=True,
    )

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255 / 2)
    synth_image = (
        synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    )
    logging.debug(f"Shape of synthesized image: {synth_image.shape}")
    source_img_name = input_img.rsplit('.', 1)[0].split('/')[-1]
    Image.fromarray(synth_image, 'RGB').save(f'{outdir}/{source_img_name}.png')
    proj_path = f'{outdir}/{source_img_name}.npz'
    np.savez(proj_path, w=projected_w.unsqueeze(0).cpu().numpy())
    return proj_path


def generate_style_frames(source_latent, style: str, output_path: str = ''):
    g_ema = Generator(1024, 512, 8, channel_multiplier=2).to('cuda')
    checkpoint = torch.load(f'./models/stylegan_nada/{style}.pt')
    g_ema.load_state_dict(checkpoint['g_ema'])

    w = torch.from_numpy(source_latent).float().cuda()
    with torch.no_grad():
        img, _ = g_ema([w], input_is_latent=True, truncation=1, randomize_noise=False)

    if not output_path:
        output_path = os.path.join(OUTPUT_PATH, 'tmp.jpg')
    utils.save_image(img, output_path, nrow=1, normalize=True, scale_each=True, range=(-1, 1))
    logging.debug(f"Saving frame to {output_path}")
    return Image.open(output_path)


def generate_gif(source_face: str, image_count: int = 5, style: str = '') -> str:
    logger.debug("Generating GIF...")
    source_img_name = source_face.rsplit('.', 1)[0].split('./images/')[1]
    source_img_proj = f"./z_output/{source_img_name}.npz"
    source_img_proj_path = Path(source_img_proj)
    edit_img_lc = f"./z_output/{source_img_name}-edit.npz"
    edit_img_lc_path = Path(edit_img_lc)

    # Generate the images for the GIF
    imgs = []
    if edit_img_lc_path.is_file():
        mlc = np.load(str(edit_img_lc_path))['x']
        input_lc = np.array(np.load(str(source_img_proj_path))['w'])
        interps = interpolate_latent_codes(input_lc, mlc, np.arange(0, 1, 1 / image_count))

        if style and style != 'none':
            for idx, interp_lc in enumerate(interps):
                img = generate_style_frames(interp_lc, style, os.path.join(OUTPUT_PATH, f'tmp_{idx}.jpg'))
                imgs.append(img.resize((512, 512)))
        else:
            for interp_lc in interps:
                imgs.append(synthesize_new_img(interp_lc).resize((512, 512)))

        # Save the GIF
        gif_path = f"{OUTPUT_PATH}/{source_img_name}-edit.gif"
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=1, loop=0, optimize=True)
        logger.debug(f"Saved GIF at {gif_path}...")

        return gif_path
    return ""


def interpolate_latent_codes(lc_1: NDArray, lc_2: NDArray, factors: NDArray):
    interps = []
    for fac in factors:
        interps.append(lc_1 * (1 - fac) + lc_2 * fac)
    return interps

from pathlib import Path
from loguru import logger
import numpy as np


def load_latent_vectors(path):
    # Reference:
    # - https://amarsaini.github.io/Epoching-Blog/jupyter/2020/08/10/Latent-Space-Exploration-with-StyleGAN2.html#6.-Latent-Directions/Controls-to-modify-our-projected-images
    # - https://twitter.com/robertluxemburg/status/1207087801344372736
    files = [x for x in Path(path).iterdir() if str(x).endswith('.npy')]
    latent_vectors = {f.name[:-4]: np.load(f) for f in files}
    return latent_vectors


def edit_image(latent_vector_info: dict, img_path: Path, feature_info: dict):
    latent_controls = latent_vector_info
    supported_attr = list(latent_controls.keys())
    logger.debug(
        f"Info about the image controls: {len(latent_controls), latent_controls.keys()}"
    )

    latent_code_for_input_img = np.array(np.load(img_path)['w'])

    for a_n in supported_attr:
        if a_n in feature_info:
            l_c = latent_controls[a_n]
            latent_code_for_input_img += l_c * feature_info.get(a_n, 0)
    return latent_code_for_input_img

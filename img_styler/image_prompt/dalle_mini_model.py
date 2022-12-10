# Reference: https://colab.research.google.com/github/borisdayma/dalle-mini/blob/main/tools/inference/inference_pipeline.ipynb#scrollTo=sOtoOmYsSYPz

import os
import toml
from pathlib import Path
from loguru import logger

import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel

# Load the config file to read in system settings.
base_path = (Path(__file__).parent / "../configs/").resolve()
app_settings = toml.load(f"{base_path}/settings.toml")


class DalleMini:
    def __init__(self):
        model_path = app_settings["PretrainedModels"]["DALLEMini"]
        vqgan_path = app_settings["PretrainedModels"]["VQGAN"]

        logger.debug("Initializing models...")
        # Define models
        self.processor = DalleBartProcessor.from_pretrained(model_path, revision=None)
        logger.debug("DALL-E Processor model initialized.")
        self.model, self.params = DalleBart.from_pretrained(
            model_path, revision=None, dtype=jnp.float16, _do_init=False
        )
        logger.debug("DALL-E mini model initialized.")


        self.vqgan, self.vqgan_params = VQModel.from_pretrained(
            vqgan_path, revision="e93a26e7707683d349bf5d5c41c5b0ef69b677a9", _do_init=False
        )
        logger.debug("VQGAN model initialized.")

    def p_generate(
        self, tokenized_prompt, key, top_k, top_p, temperature, condition_scale
    ):
        return self.model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=self.params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )

    def p_decode(self, indices):
        return self.vqgan.decode_code(indices, params=self.vqgan_params)

    def generate_image(self,
                       prompt: str,
                       output_path: str,
                       seed: int,
                       top_k=None,
                       top_p=None,
                       temperature=None,
                       condition_scale: float = 10.0):

        prompts = [prompt]
        logger.debug(f"Prompts: {prompts}\n")
        tokenized_prompt = self.processor(prompts)

        # generate images
        logger.debug("Generating encoded image...")
        encoded_images = self.p_generate(
            tokenized_prompt,
            jax.random.PRNGKey(seed),
            top_k,
            top_p,
            temperature,
            condition_scale
        )

        # remove BOS
        encoded_images = encoded_images.sequences[..., 1:]
        # decode images
        logger.debug("Decoding encoded image...")
        decoded_images = self.p_decode(encoded_images)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        img = Image.fromarray(np.asarray(decoded_images[0] * 255, dtype=np.uint8))

        filename = os.path.join(output_path, 'result.jpg')
        img.save(filename)
        logger.debug(f"Done. Saved image to {filename}")

        return filename


if __name__ == "__main__":
    dalle = DalleMini()
    prompt = "sunset over a lake in the mountains"
    image_path = dalle.generate_image(prompt, "", 0, 0.9, 0.1, 10)

# Reference: https://github.com/lllyasviel/ControlNet/blob/main/gradio_canny2image.py
import gc

import cv2
import numpy as np
import PIL.Image
import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)
from img_styler.image_prompt.control_net.annotator import (
    CannyDetector, ContentShuffleDetector, HEDdetector, LineartAnimeDetector,
    LineartDetector, MidasDetector, MLSDdetector, NormalBaeDetector,
    OpenposeDetector, PidiNetDetector)
from img_styler.image_prompt.control_net.annotator.util import (HWC3,
                                                                ade_palette)
from transformers import (AutoImageProcessor, UperNetForSemanticSegmentation,
                          pipeline)

device = "cuda" if torch.cuda.is_available() else "cpu"


class ControlNetMode:
    POSE = "pose"
    CANNY = "canny"
    MLSD = "mlsd"
    SCRIBBLE = "scribble"
    SOFTEDGE = "softedge"
    SEG = "seg"
    DEPTH = "depth"
    NORMAL = "normal"
    LINEART = "lineart"
    LINEART_ANIME = "lineart_anime"
    SHUFFLE = "shuffle"
    IP2P = "ip2p"
    INPAINT = "inpaint"


CONTROLNET_MODEL_IDS = {
    ControlNetMode.POSE: "lllyasviel/control_v11p_sd15_openpose",
    ControlNetMode.CANNY: "lllyasviel/control_v11p_sd15_canny",
    ControlNetMode.MLSD: "lllyasviel/control_v11p_sd15_mlsd",
    ControlNetMode.SCRIBBLE: "lllyasviel/control_v11p_sd15_scribble",
    ControlNetMode.SOFTEDGE: "lllyasviel/control_v11p_sd15_softedge",
    ControlNetMode.SEG: "lllyasviel/control_v11p_sd15_seg",
    ControlNetMode.DEPTH: "lllyasviel/control_v11f1p_sd15_depth",
    ControlNetMode.NORMAL: "lllyasviel/control_v11p_sd15_normalbae",
    ControlNetMode.LINEART: "lllyasviel/control_v11p_sd15_lineart",
    ControlNetMode.LINEART_ANIME: "lllyasviel/control_v11p_sd15s2_lineart_anime",
    ControlNetMode.SHUFFLE: "lllyasviel/control_v11e_sd15_shuffle",
    ControlNetMode.IP2P: "lllyasviel/control_v11e_sd15_ip2p",
    ControlNetMode.INPAINT: "lllyasviel/control_v11e_sd15_inpaint",
}


def resize_image(input_image, resolution, interpolation=None):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / max(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    if interpolation is None:
        interpolation = cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    img = cv2.resize(input_image, (W, H), interpolation=interpolation)
    return img


class DepthEstimator:
    def __init__(self):
        self.model = pipeline("depth-estimation")

    def __call__(self, image: np.ndarray, **kwargs) -> PIL.Image.Image:
        detect_resolution = kwargs.pop("detect_resolution", 512)
        image_resolution = kwargs.pop("image_resolution", 512)
        image = np.array(image)
        image = HWC3(image)
        image = resize_image(image, resolution=detect_resolution)
        image = PIL.Image.fromarray(image)
        image = self.model(image)
        image = image["depth"]
        image = np.array(image)
        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        return PIL.Image.fromarray(image)


class ImageSegmentor:
    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        self.image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

    @torch.inference_mode()
    def __call__(self, image: np.ndarray, **kwargs) -> PIL.Image.Image:
        detect_resolution = kwargs.pop("detect_resolution", 512)
        image_resolution = kwargs.pop("image_resolution", 512)
        image = HWC3(image)
        image = resize_image(image, resolution=detect_resolution)
        image = PIL.Image.fromarray(image)

        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        outputs = self.image_segmentor(pixel_values)
        seg = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(ade_palette()):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)

        color_seg = resize_image(color_seg, resolution=image_resolution, interpolation=cv2.INTER_NEAREST)
        return PIL.Image.fromarray(color_seg)


class Preprocessor:
    MODEL_ID = "lllyasviel/Annotators"

    def __init__(self):
        self.model = None
        self.name = ""

    def load(self, name: str) -> None:
        if name == self.name:
            return
        if name == "HED":
            self.model = HEDdetector.from_pretrained(self.MODEL_ID)
        elif name == "Midas":
            self.model = MidasDetector.from_pretrained(self.MODEL_ID)
        elif name == "MLSD":
            self.model = MLSDdetector.from_pretrained(self.MODEL_ID)
        elif name == "Openpose":
            self.model = OpenposeDetector.from_pretrained(self.MODEL_ID)
        elif name == "PidiNet":
            self.model = PidiNetDetector.from_pretrained(self.MODEL_ID)
        elif name == "NormalBae":
            self.model = NormalBaeDetector.from_pretrained(self.MODEL_ID)
        elif name == "Lineart":
            self.model = LineartDetector.from_pretrained(self.MODEL_ID)
        elif name == "LineartAnime":
            self.model = LineartAnimeDetector.from_pretrained(self.MODEL_ID)
        elif name == "Canny":
            self.model = CannyDetector()
        elif name == "ContentShuffle":
            self.model = ContentShuffleDetector()
        elif name == "DPT":
            self.model = DepthEstimator()
        elif name == "UPerNet":
            self.model = ImageSegmentor()
        else:
            raise ValueError
        torch.cuda.empty_cache()
        gc.collect()
        self.name = name

    def __call__(self, image: PIL.Image.Image, **kwargs) -> PIL.Image.Image:
        if self.name == "Canny":
            if "detect_resolution" in kwargs:
                detect_resolution = kwargs.pop("detect_resolution")
                image = np.array(image)
                image = HWC3(image)
                image = resize_image(image, resolution=detect_resolution)
            image = self.model(image, **kwargs)
            return PIL.Image.fromarray(image)
        elif self.name == "Midas":
            detect_resolution = kwargs.pop("detect_resolution", 512)
            image_resolution = kwargs.pop("image_resolution", 512)
            image = np.array(image)
            image = HWC3(image)
            image = resize_image(image, resolution=detect_resolution)
            image = self.model(image, **kwargs)
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            return PIL.Image.fromarray(image)
        else:
            return self.model(image, **kwargs)


def get_controlnet_image_samples(
    input_img_path,
    prompt,
    seed,
    mode=ControlNetMode.CANNY,
    output_path="",
    a_prompt="",
    n_prompt="",
    image_resolution=256,
    detect_resolution=256,
    ddim_steps=20,
    guess_mode=False,
    scale=5.0,
    low_threshold=100,
    high_threshold=200,
    value_threshold=0.1,
    distance_threshold=0.1,
):
    image = PIL.Image.open(input_img_path)
    prompt = f"{prompt}, {a_prompt}"
    preprocessor = Preprocessor()
    torch.cuda.empty_cache()
    gc.collect()

    with torch.no_grad():
        if mode == ControlNetMode.CANNY:
            preprocessor.load("Canny")
            image = preprocessor(
                image=image,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                detect_resolution=image_resolution,
            )
        elif mode == ControlNetMode.MLSD:
            preprocessor.load("MLSD")
            image = preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=detect_resolution,
                thr_v=value_threshold,
                thr_d=distance_threshold,
            )
        elif mode == ControlNetMode.SCRIBBLE:
            preprocessor.load("HED")
            image = preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=detect_resolution,
                scribble=False,
            )
        elif mode == ControlNetMode.SOFTEDGE:
            preprocessor.load("PidiNet")
            image = preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=detect_resolution,
                scribble=False,
            )
        elif mode == ControlNetMode.POSE:
            preprocessor.load("Openpose")
            image = preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=detect_resolution,
                hand_and_face=True,
            )
        elif mode == ControlNetMode.SEG:
            preprocessor.load("UPerNet")
            image = preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=detect_resolution,
            )
        elif mode == ControlNetMode.DEPTH:
            preprocessor.load("DPT")
            image = preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=detect_resolution,
            )
        elif mode == ControlNetMode.NORMAL:
            preprocessor.load("NormalBae")
            image = preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=detect_resolution,
            )
        elif mode == ControlNetMode.LINEART:
            preprocessor.load("Lineart")
            image = preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=detect_resolution,
            )
        elif mode == ControlNetMode.LINEART_ANIME:
            preprocessor.load("LineartAnime")
            image = preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=detect_resolution,
            )
        elif mode == ControlNetMode.SHUFFLE:
            preprocessor.load("ContentShuffle")
            image = preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=detect_resolution,
            )
        model_id = CONTROLNET_MODEL_IDS[mode]
        controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
        controlnet.to(device)

        # By default safety checker is enabled
        # It helps in filtering out images that could be considered offensive or harmful.
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        if device == "cuda":
            pipe.enable_xformers_memory_efficient_attention()
        pipe.to(device)
        pipe.enable_model_cpu_offload()

        torch.cuda.empty_cache()
        gc.collect()

        generator = torch.manual_seed(seed)

        out_image = pipe(
            prompt,
            num_inference_steps=ddim_steps,
            generator=generator,
            negative_prompt=n_prompt,
            image=image,
            guess_mode=guess_mode,
            guidance_scale=scale,
        ).images[0]

        out_image.save(output_path)

    return output_path

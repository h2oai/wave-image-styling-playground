import base64
import io
import os
import random
from io import BytesIO
from pathlib import Path

import cv2
import dlib
import numpy as np
from deepface import DeepFace
from h2o_wave import Q, handle_on, on, site, ui
from loguru import logger
from PIL import Image

from ..caller import apply_projection, generate_gif, generate_projection, generate_style_frames, synthesize_new_img
from ..gfpgan.inference_gfpgan import init_gfpgan, restore_image
from ..image_prompt.dalle_mini_model import DalleMini
from ..image_prompt.stable_diffusion import generate_image_with_prompt
from ..latent_editor import edit_image, load_latent_vectors
from ..utils.dataops import buf2img, get_files_in_dir, img2buf, remove_file
from .capture import capture_img, draw_boundary, html_str, js_schema
from .common import progress_generate_gif, update_controls, update_faces, update_gif, update_processed_face
from .components import get_header, get_meta, get_user_title

PRE_COMPUTED_PROJECTION_PATH = "./z_output"
OUTPUT_PATH = "./var/lib/tmp/jobs/output"
INPUT_PATH = "./images"


@on()
async def close_dialog(q: Q):
    q.page["meta"].dialog = None
    await q.page.save()


@on()
async def change_theme(q: Q):
    """Change the app from light to dark mode"""

    # Toggle dark mode
    q.user.dark_mode = not q.user.dark_mode

    # Toggle theme icon
    q.page["header"].items[0].mini_button.icon = "ClearNight" if q.user.dark_mode else "Sunny"

    # Switch theme
    q.page["meta"].theme = "winter-is-coming" if q.user.dark_mode else "ember"

    await q.page.save()


async def process(q: Q):
    logger.debug(f"Source_face {q.args.source_face}")
    logger.debug(f"Style_face {q.args.style_face}")
    logger.debug(f"Z Low {q.args.z_low}")
    logger.debug(f"Z High {q.args.z_high}")

    hash = q.args["#"]
    if q.args.task_dropdown and q.client.task_choice != q.args.task_dropdown:
        logger.info(f"Task selection: {q.args.task_dropdown}")
        q.client.task_choice = q.args.task_dropdown
        reset_edit_results(q)
    if q.args.source_face and q.client.source_face != q.args.source_face:
        q.client.source_face = q.args.source_face
        reset_edit_results(q)
    if q.args.source_style:
        q.client.source_style = q.args.source_style
    if q.args.style_face:
        q.client.style_face = q.args.style_face
    if q.args.z_low:
        q.client.z_low = int(q.args.z_low)
    if q.args.z_high:
        q.client.z_high = int(q.args.z_high)
    if q.args.generate_gif:
        await progress_generate_gif(q)
        style_type = q.client.source_style[len("style_") :]
        q.client.gif_path = generate_gif(q.client.source_face, 15, style_type)
    out_path = os.path.join(OUTPUT_PATH, "temp.png")
    if q.args.img_restoration and (q.client.processedimg or q.client.source_face):
        q.client.restorer = q.client.restorer or init_gfpgan()
        img_path = q.client.processedimg or q.client.source_face
        q.client.processedimg = out_path
        restore_image(q.client.restorer, img_path, out_path)
    if q.args.prompt_model:
        q.client.prompt_model = q.args.prompt_model
    if q.args.save_img_to_list:
        new_img_path = os.path.join(INPUT_PATH, q.args.img_name)
        if os.path.exists(new_img_path):
            q.page["meta"] = ui.meta_card(
                box="",
                notification_bar=ui.notification_bar(
                    text=f'Image by the name "{q.args.img_name}" already exists!',
                    type="error",
                    position="bottom-left",
                ),
            )
        else:
            os.rename(out_path, new_img_path)
            temp_img = Image.open(new_img_path)
            temp_img.save(os.path.join(INPUT_PATH, "portrait.jpg"))

            q.app.source_faces = get_files_in_dir(dir_path=INPUT_PATH)
            q.client.processedimg = new_img_path
            q.page["meta"] = ui.meta_card(
                box="",
                notification_bar=ui.notification_bar(
                    text="Image added to list!",
                    type="success",
                    position="bottom-left",
                ),
            )
        await q.page.save()
        del q.page["meta"]

    await update_controls(q)
    await update_faces(q)
    await update_processed_face(q)
    await update_gif(q)
    if q.args.apply:
        await apply(q)
    if hash == "help":
        await help(q)
    elif hash == "capture":
        await capture(q)
    elif q.args.upload_image_dialog:
        await upload_image_dialog(q)
    elif q.args.image_upload:
        await image_upload(q)
    elif q.args.img_capture_save:
        await img_capture_save(q)
    elif q.args.img_capture_save:
        await img_capture_done(q)
    elif q.args.change_theme:
        await change_theme(q)
    elif q.args.prompt_apply:
        await prompt_apply(q)
    await q.page.save()


@on()
async def img_capture_save(q: Q):
    logger.debug(f"Save the current image.")
    new_img = q.client.current_img
    # Dump the img
    # Save new generated img locally
    file_name = f"{INPUT_PATH}/portrait.jpg"
    pre_computed_wts = f"{PRE_COMPUTED_PROJECTION_PATH}/portrait.npz"
    logger.debug(f"Image path: {file_name}")
    # Delete previous image exists.
    remove_file(file_name)
    # Also delete pre-computed latent weights
    remove_file(pre_computed_wts)

    if new_img:
        logger.debug(f"Save file on disk.")
        buf2img(new_img, file_name)
    # Refresh the page.
    q.args.img_capture_save = False
    # Update the source image cache
    q.app.source_faces = get_files_in_dir(dir_path=INPUT_PATH)
    # Set the current source face as the captured image.
    q.app.source_face = q.client.source_face = file_name
    # Return to home page.
    q.page["meta"].redirect = "/"
    await q.page.save()


@on()
async def img_capture_done(q: Q):
    logger.debug(f"Exit image capture.")
    q.page["meta"].redirect = "/"
    await q.page.save()


@on()
async def home(q: Q):
    q.page.drop()
    q.page["meta"] = get_meta(q)
    q.page["header"] = get_header(q)
    q.page["user_title"] = get_user_title(q)
    await update_controls(q)
    await update_faces(q)
    await q.page.save()


def source_face_check(q: Q, source_face_arg: str) -> bool:
    return source_face_arg != q.client.source_face


def reset_edit_results(q: Q):
    q.client.processedimg = None
    q.client.gif_path = ""


@on("source_face", source_face_check)
async def source_face(q: Q):
    logger.debug("Calling source_face")
    await process(q)


@on("task_dropdown")
async def on_task_selection(q: Q):
    logger.info("Selecting task choice")
    await process(q)


def style_face_check(q: Q, style_face_arg: str) -> bool:
    return style_face_arg != q.client.style_face


@on("style_face", style_face_check)
async def style_face(q: Q):
    logger.debug("Calling style_face")
    await process(q)


def z_low_check(q: Q, z_low_arg: str) -> bool:
    return int(z_low_arg) != q.client.z_low


@on("z_low", z_low_check)
async def z_low(q: Q):
    logger.debug("Calling z_low")
    await process(q)


def z_high_check(q: Q, z_high_arg: str) -> bool:
    return int(z_high_arg) != q.client.z_high


@on("z_high", z_high_check)
async def z_high(q: Q):
    logger.debug("Calling z_high")
    await process(q)


@on()
async def upload_image_dialog(q: Q):
    q.page["meta"].dialog = ui.dialog(
        title="Upload Image",
        closable=True,
        items=[ui.file_upload(name="image_upload", label="Upload")],
    )

    await q.page.save()


@on()
async def image_upload(q: Q):
    q.page.drop()
    q.page["meta"] = get_meta(q)
    q.page["header"] = get_header(q)
    q.page["user_title"] = get_user_title(q)
    if q.args.image_upload:
        local_path = await q.site.download(q.args.image_upload[0], "./images/")
        encoded = base64.b64encode(open(local_path, "rb").read()).decode("ascii")
        _img = "data:image/png;base64,{}".format(encoded)
        q.client.current_img = _img
        facial_feature_analysis(q, local_path, "Uploaded Image")
    await q.page.save()


def check_input_value(value: str, val_type, default=None):
    try:
        return val_type(value)
    except ValueError:
        return default


@on("prompt_apply")
async def prompt_apply(q: Q):
    logger.info("Enable prompt.")
    logger.info(f"Prompt value: {q.args.prompt_textbox}")
    random_seed = random.randint(600000000000000, 700000000000000)

    random_seed = q.args.prompt_seed = check_input_value(q.args.prompt_seed, int, random_seed)
    no_images = q.args.no_images = check_input_value(q.args.no_images, int, 1)

    if q.client.prompt_model == "prompt_sd":
        logger.info(f"Number of steps: {q.args.diffusion_n_steps}")
        logger.info(f"Guidance scale: {q.args.prompt_guidance_scale}")
        logger.info(f"Sampler choice: {q.args.df_sampling_dropdown}")
        if q.args.prompt_use_source_img:
            res_path = generate_image_with_prompt(
                input_img_path=q.client.source_face,
                prompt_txt=q.args.prompt_textbox,
                negative_prompt=q.args.negative_prompt_textbox,
                n_steps=q.args.diffusion_n_steps,
                guidance_scale=q.args.prompt_guidance_scale,
                sampler_type=q.args.df_sampling_dropdown,
                output_path=OUTPUT_PATH,
                seed=random_seed,
            )
        else:  # Don't initialize with source image
            res_path = generate_image_with_prompt(
                prompt_txt=q.args.prompt_textbox,
                negative_prompt=q.args.negative_prompt_textbox,
                n_steps=q.args.diffusion_n_steps,
                guidance_scale=q.args.prompt_guidance_scale,
                sampler_type=q.args.df_sampling_dropdown,
                output_path=OUTPUT_PATH,
                seed=random_seed,
                n_images=no_images,
            )
        q.client.negative_prompt_textbox = q.args.negative_prompt_textbox
        q.client.diffusion_n_steps = q.args.diffusion_n_steps
        q.client.prompt_guidance_scale = q.args.prompt_guidance_scale
        q.client.prompt_use_source_img = q.args.prompt_use_source_img
    else:
        logger.info(f"Top-K: {q.args.prompt_top_k}")
        logger.info(f"Top-P: {q.args.prompt_top_p}")
        logger.info(f"Temperature: {q.args.prompt_temp}")
        logger.info(f"Condition Scale: {q.args.prompt_cond_scale}")
        dalle_mini_obj = DalleMini()
        res_path = dalle_mini_obj.generate_image(
            prompt=q.args.prompt_textbox,
            output_path=OUTPUT_PATH,
            seed=random_seed,
            top_k=check_input_value(q.args.prompt_top_k, int),
            top_p=check_input_value(q.args.prompt_top_p, float),
            temperature=check_input_value(q.args.prompt_temp, float),
            condition_scale=q.args.prompt_cond_scale,
        )

        q.client.prompt_top_k = q.args.prompt_top_k
        q.client.prompt_top_p = q.args.prompt_top_p
        q.client.prompt_temp = q.args.prompt_temp
        q.client.prompt_cond_scale = q.args.prompt_cond_scale

    q.client.prompt_textbox = q.args.prompt_textbox
    q.client.prompt_seed = int(q.args.prompt_seed)
    q.client.no_images = int(q.args.no_images)
    q.client.processedimg = res_path
    await update_processed_face(q)


@on("#help")
async def help(q: Q):
    logger.info("Help")
    url = "https://github.com/h2oai/wave-image-styler/blob/main/style_engineering_ebook.md"
    q.page["meta"].script = ui.inline_script(
        f"""
        href = window.location.href;
        window.open('{url}', '_blank').focus();
        window.location.href = href.replace('#help', '');
        """
    )
    await q.page.save()


@on("#capture")
async def capture(q: Q):
    if q.args.img_capture_save:
        await img_capture_save(q)
    else:
        logger.debug("Capture clicked.")
        q.page.drop()
        q.page["meta"] = get_meta(q)
        q.page["header"] = get_header(q)
        q.page["user_title"] = get_user_title(q)
        _img = await capture_img(q)
        q.client.current_img = _img

        if _img:
            # Once the image is captured
            # Lets do some lightweight Facial Attribute analysis, namely emotions
            facial_feature_analysis(q, _img)
        elif q.args.exit_camera:
            # Return to home page.
            q.page["meta"].redirect = "/"
            await q.page.save()
        else:
            q.page["meta"].script = ui.inline_script(content=js_schema, requires=[], targets=["video"])
            q.page["plot"] = ui.markup_card(
                box=ui.box("middle_left", order=2, height="950px", width="950px"),
                title="",
                content=html_str,
            )
            # TODO Replace css styling
            q.page["meta"].stylesheets = [
                ui.stylesheet(path="https://cdn.jsdelivr.net/npm/bootstrap@5.1.0/dist/css/bootstrap.min.css")
            ]

    await q.page.save()


def rotate_face(image_path: str):
    # face = dlib.load_rgb_image(image_path)
    face = Image.open(image_path)
    face = np.array(face.convert("RGB"))
    _img = cv2.rotate(face, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(image_path, _img)
    # Update default path as well
    default_path = f"{image_path.rsplit('/', 1)[0]}/portrait.jpg"
    cv2.imwrite(default_path, _img)
    # buff = BytesIO()
    # pil_img = Image.fromarray(_img)
    # pil_img.save(buff, format="JPEG")
    # new_image_encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
    # new_img = f"data:image/png;base64,{new_image_encoded}"


def facial_feature_analysis(q: Q, img_path: str, title="Clicked Image"):
    models = {}
    models["emotion"] = DeepFace.build_model("Emotion")
    # MTCNN (performed better than RetinaFace for the sample images tried).
    # If face is not detected; it's probably b'cauz of orientation
    # Naive approach:
    # Try rotating the image by 90 degrees to find face
    # Rotate -> ['Left', 'Right', 'Up', 'Down']
    obj = None
    for _ in range(4):
        try:
            obj = DeepFace.analyze(
                img_path=img_path,
                models=models,
                actions=["emotion"],
                detector_backend="mtcnn",
            )
            if obj and len(obj) > 0:
                break
        except ValueError as ve:
            logger.info(f"Face re-orientation might be needed.")
            new_img = rotate_face(img_path)
            # q.client.current_img = new_img
            pass
    new_img = img2buf(img_path)
    q.client.current_img = new_img
    new_image_encoded = None
    logger.info(f"Facial Attributes: {obj}")
    img_format = "data:image/png;base64,"
    if obj:
        # if face is detected
        dominant_emotion = obj["dominant_emotion"]
        logger.info(f"Dominant emotion: {dominant_emotion}")
        # Draw bounding box around the face
        _img = q.client.current_img
        # _img = _im.split(",")[1]
        base64_decoded = base64.b64decode(_img)
        image = Image.open(io.BytesIO(base64_decoded))
        img_np = np.array(image)

        x = obj["region"]["x"]
        y = obj["region"]["y"]
        w = obj["region"]["w"]
        h = obj["region"]["h"]
        img_w_box2 = draw_boundary(img_np, x, y, w, h, text=dominant_emotion)
        pil_img = Image.fromarray(img_w_box2)

        buff = BytesIO()
        pil_img = pil_img.convert("RGB")
        pil_img.save(buff, format="JPEG")
        new_image_encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
    else:
        # else proceed without as a non-portrait image
        new_image_encoded = new_img
    # Update image
    new_image = img_format + new_image_encoded

    q.page["capture_img"] = ui.form_card(
        box=ui.box("middle_left"),
        title=title,
        items=[
            ui.image("Captured Image", path=new_image, width="550px"),
            ui.buttons(
                items=[
                    ui.button("img_capture_save", "Save & Exit", icon="Save"),
                    ui.button("img_capture_done", "Ignore & Exit", icon="ChromeClose"),
                ]
            ),
        ],
    )


@on()
async def apply(q: Q):
    source_face = q.client.source_face
    style_face = q.client.style_face
    z_low = int(q.client.z_low)
    z_high = int(q.client.z_high)
    logger.debug(f"Selected source: {source_face}")
    logger.debug(f"Style source: {style_face}")
    logger.debug(f"Other values: {z_low}/{z_high}")

    # Use pre-computed projections
    source_img_name = source_face.rsplit(".", 1)[0].split("./images/")[1]
    style_img_name = style_face.rsplit(".", 1)[0].split("./images/")[1]
    source_img_proj = f"{PRE_COMPUTED_PROJECTION_PATH}/{source_img_name}.npz"
    style_img_proj = f"{PRE_COMPUTED_PROJECTION_PATH}/{style_img_name}.npz"
    source_img_proj_path = Path(source_img_proj)
    style_img_proj_path = Path(style_img_proj)

    new_img = None
    style_type = ""
    file_name = ""
    if q.client.task_choice == "C":  # Image Editing
        q.client.source_style = q.args.source_style or "style_none"
        q.client.age_slider = q.args.age_slider if q.args.age_slider else 0
        q.client.eye_distance = q.args.eye_distance if q.args.eye_distance else 0
        q.client.eyebrow_distance = q.args.eyebrow_distance if q.args.eyebrow_distance else 0
        q.client.eye_ratio = q.args.eye_ratio if q.args.eye_ratio else 0
        q.client.eyes_open = q.args.eyes_open if q.args.eyes_open else 0
        q.client.gender = q.args.gender if q.args.gender else 0

        q.client.lip_ratio = q.args.lip_ratio if q.args.lip_ratio else 0
        q.client.mouth_open = q.args.mouth_open if q.args.mouth_open else 0
        q.client.mouth_ratio = q.args.mouth_ratio if q.args.mouth_ratio else 0
        q.client.nose_mouth_distance = q.args.nose_mouth_distance if q.args.nose_mouth_distance else 0
        q.client.nose_ratio = q.args.nose_ratio if q.args.nose_ratio else 0
        q.client.nose_tip = q.args.nose_tip if q.args.nose_tip else 0
        q.client.pitch = q.args.pitch if q.args.pitch else 0
        q.client.roll = q.args.roll if q.args.roll else 0
        q.client.smile = q.args.smile if q.args.smile else 0
        q.client.yaw = q.args.yaw if q.args.yaw else 0
        latent_info = load_latent_vectors("./models/stylegan2_attributes/")

        # Update feature info for image editing
        # Dictionary format:
        # {
        #   "feature_name": "value"
        #  }
        f_i = {
            "age": q.client.age_slider,
            "eye_distance": q.client.eye_distance,
            "eye_eyebrow_distance": q.client.eyebrow_distance,
            "eye_ratio": q.client.eye_ratio,
            "eyes_open": q.client.eyes_open,
            "gender": q.client.gender,
            "lip_ratio": q.client.lip_ratio,
            "mouth_open": q.client.mouth_open,
            "mouth_ratio": q.client.mouth_ratio,
            "nose_mouth_distance": q.client.nose_mouth_distance,
            "nose_ratio": q.client.nose_ratio,
            "nose_tip": q.client.nose_tip,
            "pitch": q.client.pitch,
            "roll": q.client.roll,
            "smile": q.client.smile,
            "yaw": q.client.yaw,
        }

        style_type = q.client.source_style[len("style_") :]
        logger.debug(f"Source style: {style_type}")
        if source_img_proj_path.is_file() and style_type == "none":
            mlc = edit_image(
                latent_info,
                source_img_proj_path,
                f_i,
            )
            new_img = synthesize_new_img(mlc)
        else:
            if not source_img_proj_path.is_file():
                generate_projection(source_face, PRE_COMPUTED_PROJECTION_PATH)
                logger.info(f"New projections computed.")
                source_img_proj = f"{PRE_COMPUTED_PROJECTION_PATH}/{source_img_name}.npz"
                source_img_proj_path = Path(source_img_proj)

            mlc = edit_image(latent_info, source_img_proj_path, f_i)
            if style_type != "none":
                file_name = OUTPUT_PATH + f"/{source_img_name}.jpg"
                new_img = generate_style_frames(mlc, style_type, file_name)
            else:
                new_img = synthesize_new_img(mlc)
        if mlc is not None:
            edit_img_lc = f"{PRE_COMPUTED_PROJECTION_PATH}/{source_img_name}-edit.npz"
            logger.debug(f"Saving to {edit_img_lc}")
            edit_img_lc_path = Path(edit_img_lc)
            np.savez(edit_img_lc_path, x=mlc)
    elif q.client.task_choice == "B":  # Image Styling
        # Check if precomputed latent space for source img exists
        if source_img_proj_path.is_file() & style_img_proj_path.is_file():
            swap_idxs = (z_low, z_high)
            new_projection = apply_projection(source_img_proj, style_img_proj, swap_idxs)
            new_img = synthesize_new_img(new_projection)
        else:
            if not source_img_proj_path.is_file():
                generate_projection(source_face, PRE_COMPUTED_PROJECTION_PATH)
                logger.info(f"New projections computed.")
                source_img_proj = f"{PRE_COMPUTED_PROJECTION_PATH}/{source_img_name}.npz"
                source_img_proj_path = Path(source_img_proj)
            if not style_img_proj_path.is_file():
                generate_projection(style_face, PRE_COMPUTED_PROJECTION_PATH)
                logger.info(f"New projections computed.")
                style_img_proj = f"{PRE_COMPUTED_PROJECTION_PATH}/{style_img_name}.npz"
                style_img_proj_path = Path(style_img_proj)

            if source_img_proj_path.is_file() & style_img_proj_path.is_file():
                swap_idxs = (z_low, z_high)
                new_projection = apply_projection(source_img_proj, style_img_proj, swap_idxs)
                new_img = synthesize_new_img(new_projection)

    # Save new generated img locally
    if not file_name:
        file_name = f"{OUTPUT_PATH}/{source_img_name}_{style_img_name}_{z_low}-{z_high}.jpg"
        logger.debug(f"Generate img: {file_name}")
        if new_img:
            new_img.save(file_name)

    reset_edit_results(q)
    q.client.processedimg = file_name
    # Update the page with processed image.
    await update_controls(q)
    await update_processed_face(q, save=True)
    await update_gif(q)

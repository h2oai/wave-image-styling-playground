import os
from typing import Optional

from h2o_wave import Q, ui

from ..utils.dataops import img2buf
from .layouts import get_layouts


def get_meta(q: Q):
    return ui.meta_card(
        box="",
        title="ImageStyler",
        layouts=get_layouts(),
        theme="winter-is-coming" if q.user.dark_mode else "ember",
    )


def get_header(q: Q):
    return ui.header_card(
        box="header",
        title="Image Styling Art Studio",
        subtitle="Style your images and have fun",
        image=q.app.logo_path,
        items=[
            # This button will toggle the dark mode
            ui.mini_button(
                # A handler (an async function with @on() decorator) must be
                # defined to handle the button click event
                name="change_theme",
                icon="ClearNight" if q.user.dark_mode else "Sunny",
                label="",
            ),
        ],
        color="transparent",
        nav=[
            ui.nav_group(
                "Menu",
                items=[
                    ui.nav_item(name="#", label="Art Studio", icon="FormatPainter"),
                    ui.nav_item(name="#help", label="Help", icon="help"),
                ],
            ),
        ],
    )


def get_footer():
    return ui.footer_card(
        box=ui.box("footer", order=1),
        caption=(
            '<p>Made with 💛️ using <a href="https://h2oai.github.io/wave/",'
            ' target="_blank">Wave</a>. (c) 2022-23 H2O.ai. All rights reserved.</p>'
        ),
    )


def get_user_title(q: Q):
    return ui.section_card(
        box=ui.box(zone="title", order=1),
        title="",
        subtitle="",
    )


def get_controls(q: Q):
    task_choices = [
        ui.choice("A", "Image Restoration"),
        ui.choice("B", "Image Styling"),
        ui.choice("C", "Image Editing"),
        ui.choice("D", "Image Prompt"),
    ]
    task_choice_dropdown = ui.dropdown(
        name="task_dropdown",
        label="Select a styling option",
        value=q.client.task_choice,
        required=True,
        trigger=True,
        choices=task_choices,
        tooltip="There are few options available. \
            Image Restoration (Increase resolution and fix artifacts in an existing image), \
            Image Styling (Transfer a style to an original image), \
            Image Editing (Edit and transform an existing image), and \
            Image Prompt (Generate image via prompt)",
    )
    landmark_controls = [
        ui.separator(label="Modify"),
        ui.slider(
            name="age_slider",
            label="Age",
            min=-10,
            max=10,
            step=1,
            value=q.client.age_slider if q.client.age_slider else 0,
        ),
        ui.slider(
            name="eye_distance",
            label="Eye Distance",
            min=-10,
            max=10,
            step=1,
            value=q.client.eye_distance if q.client.eye_distance else 0,
        ),
        ui.slider(
            name="eyebrow_distance",
            label="Eyebrow Distance",
            min=-10,
            max=10,
            step=1,
            value=q.client.eyebrow_distance if q.client.eyebrow_distance else 0,
        ),
        ui.slider(
            name="eye_ratio",
            label="Eye Ratio",
            min=-10,
            max=10,
            step=1,
            value=q.client.eye_ratio if q.client.eye_ratio else 0,
        ),
        ui.slider(
            name="eyes_open",
            label="Eyes Open",
            min=-10,
            max=10,
            step=1,
            value=q.client.eyes_open if q.client.eyes_open else 0,
        ),
        ui.slider(
            name="gender",
            label="Gender",
            min=-10,
            max=10,
            step=1,
            value=q.client.gender if q.client.gender else 0,
        ),
        ui.slider(
            name="lip_ratio",
            label="Lip Ratio",
            min=-10,
            max=10,
            step=1,
            value=q.client.lip_ratio if q.client.lip_ratiolip_ratio else 0,
        ),
        ui.slider(
            name="mouth_open",
            label="Mouth Open",
            min=-10,
            max=10,
            step=1,
            value=q.client.mouth_open if q.client.mouth_open else 0,
        ),
        ui.slider(
            name="mouth_ratio",
            label="Mouth Ratio",
            min=-10,
            max=10,
            step=1,
            value=q.client.mouth_ratio if q.client.mouth_ratio else 0,
        ),
        ui.slider(
            name="nose_mouth_distance",
            label="Nose Mouth Distance",
            min=-10,
            max=10,
            step=1,
            value=q.client.nose_mouth_distance if q.client.nose_mouth_distance else 0,
        ),
        ui.slider(
            name="nose_ratio",
            label="Nose Ratio",
            min=-10,
            max=10,
            step=1,
            value=q.client.nose_ratio if q.client.nose_ratio else 0,
        ),
        ui.slider(
            name="nose_tip",
            label="Nose Tip",
            min=-10,
            max=10,
            step=1,
            value=q.client.nose_tip if q.client.nose_tip else 0,
        ),
        ui.slider(
            name="pitch",
            label="Pitch",
            min=-10,
            max=10,
            step=1,
            value=q.client.pitch if q.client.pitch else 0,
        ),
        ui.slider(
            name="roll",
            label="Roll",
            min=-10,
            max=10,
            step=1,
            value=q.client.roll if q.client.roll else 0,
        ),
        ui.slider(
            name="smile",
            label="Smile",
            min=-10,
            max=10,
            step=1,
            value=q.client.smile if q.client.smile else 0,
        ),
        ui.slider(
            name="yaw",
            label="Yaw",
            min=-10,
            max=10,
            step=1,
            value=q.client.yaw if q.client.yaw else 0,
        ),
    ]
    if q.client.task_choice == "A":  # Image Restoration
        img_name_parts = q.client.source_face.split("/")[-1].split(".")
        new_img_name = ".".join(img_name_parts[:-1]) + "_fixed." + img_name_parts[-1]
        disabled = q.client.processedimg is None
        return ui.form_card(
            box=ui.box(zone="side_controls", order=1),
            items=[
                task_choice_dropdown,
                ui.dropdown(
                    name="source_face",
                    label="Source Image",
                    choices=[ui.choice(name=x, label=os.path.basename(x)) for x in q.app.source_faces],
                    value=q.client.source_face,
                    tooltip="Select a source image to be enhanced.",
                    trigger=True,
                ),
                ui.buttons(
                    [
                        ui.button(name="upload_image_dialog", label="Upload", primary=True),
                        ui.button(name="#capture", label="Capture", primary=True),
                    ],
                    justify="end",
                ),
                ui.buttons(
                    [ui.button("img_restoration", "Apply", primary=True)],
                    justify="end",
                ),
                ui.separator(),
                ui.textbox(
                    "img_name",
                    "Save fixed image for future use",
                    value=new_img_name,
                    disabled=disabled,
                ),
                ui.buttons(
                    [ui.button("save_img_to_list", "Save", primary=True, disabled=disabled)],
                    justify="end",
                ),
            ],
        )
    elif q.client.task_choice == "B":
        # Includes styling controls.
        return ui.form_card(
            box=ui.box(
                zone="side_controls",
                order=1,
            ),
            items=[
                task_choice_dropdown,
                ui.dropdown(
                    name="source_face",
                    label="Source Image",
                    choices=[ui.choice(name=x, label=os.path.basename(x)) for x in q.app.source_faces],
                    value=q.client.source_face,
                    tooltip="Select a source image. One can upload a new source image as well.",
                    trigger=True,
                ),
                ui.dropdown(
                    name="style_face",
                    label="Style Face",
                    choices=[ui.choice(name=x, label=os.path.basename(x)) for x in q.app.style_faces],
                    value=q.client.style_face,
                    tooltip="Select a style face from the provided options.",
                    trigger=True,
                ),
                ui.dropdown(
                    name="z_low",
                    label="Z Low",
                    choices=[ui.choice(name=str(x), label=str(x)) for x in range(0, 17)],
                    value=str(q.client.z_low),
                    tooltip="Latent space range (0-16)",
                    trigger=True,
                ),
                ui.dropdown(
                    name="z_high",
                    label="Z High",
                    choices=[ui.choice(name=str(x), label=str(x)) for x in range(0, 17)],
                    value=str(q.client.z_high),
                    tooltip="Latent space range (0-16)",
                    trigger=True,
                ),
                ui.text_m("**Recommended pairs for Z Low/Z High:**"),
                ui.text_s("(1, 3), (3, 4), (4, 5), (4, 6), (3, 14), (1, 8)"),
                ui.buttons(
                    [
                        ui.button(
                            name="upload_image_dialog",
                            label="Upload",
                            primary=True,
                            tooltip="Upload an image.",
                        ),
                        ui.button(
                            name="#capture",
                            label="Capture",
                            primary=True,
                            tooltip="Upload an image using the camera.",
                        ),
                    ],
                    justify="end",
                ),
                ui.buttons(
                    [
                        ui.button(name="apply", label="Apply", primary=True),
                    ],
                    justify="end",
                ),
            ],
        )
    elif q.client.task_choice == "C":
        style_names = {
            "none": "None",
            "anime": "Anime",
            "botero": "Botero",
            "crochet": "Crochet",
            "cubism": "Cubism",
            "disney_princess": "Disney Princess",
            "edvard_munch": "Edvard Munch",
            "elf": "Elf",
            "ghibli": "Ghibli",
            "grafitti_on_wall": "Grafitti on Wall",
            "groot": "Groot",
            "joker": "Joker",
            "marble": "Marble",
            "modernism": "Modernism",
            "modigliani": "Modigliani",
            "mona_lisa": "Mona Lisa",
            "oil": "Oil",
            "pixar": "Pixar",
            "plastic_puppet": "Plastic Puppet",
            "rick_morty": "Rick and Morty",
            "shrek": "Shrek",
            "simpson": "Simpson",
            "sketch": "Sketch",
            "ssj": "Super Saiyan",
            "thanos": "Thanos",
            "ukiyoe": "Ukiyoe",
            "van_gogh": "Van Gogh",
            "vintage_comics": "Vintage Comics",
            "werewolf": "Werewolf",
            "white_walker": "White Walker",
            "witcher": "Witcher",
            "zombie": "Zombie",
            "zuckerberg": "Zuckerberg",
        }
        # Includes edit controls
        edit_controls = ui.form_card(
            box=ui.box(
                zone="side_controls",
                order=1,
            ),
            items=[
                task_choice_dropdown,
                ui.dropdown(
                    name="source_face",
                    label="Source Image",
                    choices=[ui.choice(name=x, label=os.path.basename(x)) for x in q.app.source_faces],
                    value=q.client.source_face,
                    trigger=True,
                    tooltip="Select a source image for editing. One can upload a new source image as well.",
                ),
                ui.dropdown(
                    name="source_style",
                    label="Styles",
                    choices=[ui.choice(name=f"style_{x}", label=style_names[x]) for x in style_names],
                    value=q.client.source_style or "style_none",
                    tooltip="Select a pre-configured style to adapt.",
                    trigger=True,
                ),
                ui.buttons(
                    [
                        ui.button(name="upload_image_dialog", label="Upload", primary=True),
                        ui.button(name="#capture", label="Capture", primary=True),
                    ],
                    justify="end",
                ),
                ui.buttons(
                    [
                        ui.button(name="apply", label="Apply", primary=True),
                        ui.button(
                            name="generate_gif",
                            label="Generate GIF",
                            primary=True,
                            disabled=(q.client.processedimg is None),
                        ),
                    ],
                    justify="end",
                ),
            ],
        )
        start_index = 3
        [edit_controls.items.insert(start_index + _index, _item) for _index, _item in enumerate(landmark_controls)]
        return edit_controls
    else:  # Option: 'D' Image Prompt
        return ui.form_card(
            box=ui.box(zone="side_controls", order=1),
            items=[
                ui.dropdown(
                    name="task_dropdown",
                    label="Select a styling option",
                    value=q.client.task_choice,
                    required=True,
                    trigger=True,
                    choices=task_choices,
                    tooltip="There are few options available. \
                        Image Styling (Transfer a style to an original image), \
                        Image Editing (Edit and transform an existing image), and \
                        Image Restoration (Increase resolution and fix artifacts in an existing image), and \
                        Image Prompt (Generate image via prompt)",
                ),
                ui.separator(),
                ui.dropdown(
                    name="prompt_model",
                    label="Model",
                    choices=[
                        ui.choice(name="prompt_sd", label="Stable Diffusion"),
                        ui.choice(name="prompt_dalle_mini", label="DALL-E mini"),
                        ui.choice(name="prompt_controlnet", label="ControlNet"),
                    ],
                    value=q.client.prompt_model,
                    trigger=True,
                    tooltip="Select a model to use for prompting.",
                ),
            ]
            + (
                [
                    ui.choice_group(
                        name="choice_group_prompt",
                        label="Options",
                        value="checkbox_without_training",
                        choices=[
                            ui.choice(name="checkbox_without_training", label="Default diffusion"),
                            ui.choice(
                                name="checkbox_re_training",
                                label="Dreambooth fine-tuning",
                                disabled=True,
                            ),
                        ],
                        tooltip="Stable Diffusion is a text-to-image latent diffusion model created by the researchers and engineers from CompVis, Stability AI and LAION.",
                    ),
                ]
                if q.client.prompt_model == "prompt_sd"
                else [
                    ui.dropdown(
                        name="source_face",
                        label="Source Image",
                        choices=[ui.choice(name=x, label=os.path.basename(x)) for x in q.app.source_faces],
                        value=q.client.source_face,
                        trigger=True,
                        tooltip="Select a source image for editing. One can upload a new source image as well.",
                    ),
                    ui.choice_group(
                        name="choice_group_prompt",
                        label="Options",
                        value="checkbox_canny",
                        choices=[
                            ui.choice(name="checkbox_canny", label="Canny2Image"),
                            ui.choice(name="checkbox_scribble", label="Scribble2Image", disabled=True),
                        ],
                        tooltip="Stable Diffusion is a text-to-image latent diffusion model created by the researchers and engineers from CompVis, Stability AI and LAION.",
                    ),
                    ui.buttons(
                        [
                            ui.button(name="upload_image_dialog", label="Upload", primary=True),
                        ],
                        justify="end",
                    ),
                ]
                if q.client.prompt_model == "prompt_controlnet"
                else []
            ),
        )


def get_source_header():
    return ui.section_card(
        box=ui.box(zone="middle_left", order=1, height="45px"),
        title="Source Image",
        subtitle="",
        items=[
            ui.dropdown(
                name="source_face",
                label="",
                choices=[ui.choice(name=x, label=x) for x in ["one.jpg", "two.jpg", "three.jpg"]],
                value="one.jpg",
                trigger=True,
            )
        ],
    )


def get_style_header():
    return ui.section_card(
        box=ui.box(zone="middle_right", order=1, height="45px"),
        title="Style Face",
        subtitle="",
        items=[
            ui.dropdown(
                name="style_face",
                label="",
                choices=[ui.choice(name=x, label=x) for x in ["one.jpg", "two.jpg", "three.jpg"]],
                value="one.jpg",
                trigger=True,
            )
        ],
    )


def get_source_face_card(image, type, height: str = "420px", width: str = "400px"):
    return ui.image_card(
        box=ui.box("middle_left", order=2, height=height, width=width),
        title="Source Image",
        type=type,
        image=image,
    )


def get_style_face_card(image, type):
    return ui.image_card(
        box=ui.box("middle_right", order=2, height="420px", width="400px"),
        title="Style Face",
        type=type,
        image=image,
    )


def clear_grid_view(q):
    if q.client.n_cards:
        for _index in range(int(q.client.n_cards)):
            del q.page[f"img_card_{_index}"]


def display_grid_view(q: Q, image_paths: Optional[list] = None, titles: Optional[list] = None, type="jpg"):
    n_rows = len(image_paths) if image_paths else 1
    if n_rows == 1:
        _height = 600
        _width = 600
    else:
        _height = 400
        _width = 400
    q.client.n_cards = n_rows
    if image_paths:
        for _index in range(n_rows):
            _img = img2buf(image_paths[_index])
            q.page[f"img_card_{_index}"] = ui.image_card(
                box=ui.box("bottom", order=1, height=f"{_height}px", width=f"{_width}px"),
                title=titles[_index],
                type=type,
                image=_img,
            )
    else:
        ui.form_card(
            box=ui.box("bottom", order=1, height=_height, width=_width),
            title="Generated Images",
            items=[ui.separator(), ui.text("'Draw' to generate a new images!")],
        )


def get_processed_face_card(
    image,
    type,
    layout_pos="main",
    order=1,
    title="Synthesized Image",
    empty_msg: str = "'Apply' to generate a new face!",
    height="520px",
    width="500px",
):
    if image:
        return ui.image_card(
            box=ui.box(layout_pos, order=order, height=height, width=width),
            title=title,
            type=type,
            image=image,
        )
    else:
        return ui.form_card(
            box=ui.box(layout_pos, order=order, height=height, width=width),
            title=title,
            items=[ui.separator(), ui.text(empty_msg)],
        )


def get_generate_gif_progress_card(layout_pos="main", order=1, title="Generated GIF", height="520px", width="500px"):
    return ui.form_card(
        box=ui.box(layout_pos, order=order, height=height, width=width),
        title=title,
        items=[ui.progress(label="Generating GIF...")],
    )


async def get_gif_generated_card(q: Q, gif, order=1, title="Generated GIF"):
    download_path = await q.site.upload([q.client.gif_path])
    return ui.form_card(
        box=ui.box("main", order=order, height="550px", width="500px"),
        title=title,
        items=[
            ui.image("GIF", "gif", gif, width="465px"),
            ui.link(label=f"Download GIF", download=True, path=download_path[0]),
        ],
    )

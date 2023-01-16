import base64
import os
import random

from h2o_wave import Q, ui

from ..utils.dataops import img2buf
from .components import (
    display_grid_view,
    clear_grid_view,
    get_controls,
    get_footer,
    get_generate_gif_progress_card,
    get_gif_generated_card,
    get_header,
    get_meta,
    get_processed_face_card,
    get_source_face_card,
    get_style_face_card,
    get_user_title,
)


async def update_faces(q: Q, save=False):
    del q.page["prompt_form"]
    if not q.client.source_face or not os.path.exists(q.client.source_face):
        q.client.source_face = random.choice(q.app.source_faces)

    if q.client.task_choice == "B":
        q.page["source_face"] = get_source_face_card(img2buf(q.client.source_face), type="jpg")
    if q.client.task_choice != "D":  # For A, C
        q.page["source_face"] = get_source_face_card(
            img2buf(q.client.source_face), type="jpg", height="520px", width="500px"
        )
    if q.client.task_choice == "D":
        txt_val = q.client.prompt_textbox if q.client.prompt_textbox else ""
        del q.page["source_face"]
        q.page["prompt_form"] = ui.form_card(
            ui.box("main", order=1, height="200px", width="900px"),
            items=[
                ui.copyable_text(
                    name="prompt_textbox",
                    label="Prompt (Express your creativity)",
                    multiline=True,
                    value=txt_val,
                ),
                ui.button(name="prompt_apply", label="Draw"),
            ],
        )

    del q.page["style_face"]
    if q.client.task_choice == "B":
        q.page["style_face"] = get_style_face_card(img2buf(q.client.style_face), type="jpg")
    if save:
        await q.page.save()


async def update_processed_face(q: Q, save=False):
    # Delete pages not needed.
    # It's cheap to create them.
    del q.page["processed_face"]
    del q.page["prompt_form"]
    clear_grid_view(q)

    if q.client.task_choice == "B":
        img_buf = img2buf(q.client.processedimg) if q.client.processedimg else None
        q.page["processed_face"] = get_processed_face_card(img_buf, type="jpg")
    else:
        if q.client.task_choice != "D":
            img_buf = img2buf(q.client.processedimg) if q.client.processedimg else None
            q.page["processed_face"] = get_processed_face_card(
                img_buf,
                title="Generated Image",
                type="jpg",
                layout_pos="middle_right",
                order=2,
            )
        if q.client.task_choice == "D":
            if q.client.prompt_model == "prompt_sd":
                items = [
                    ui.inline(
                        items=[
                            ui.button(name="prompt_apply", label="Draw"),
                        ]
                    )
                ]
                extra_settings = [
                    ui.textbox(
                        name="negative_prompt_textbox",
                        label="Do not include",
                        multiline=True,
                        value=q.client.negative_prompt_textbox,
                        tooltip="Helpful for fixing image abnormalities",
                    ),
                    ui.expander(
                        name="expander",
                        label="Settings",
                        items=[
                            ui.textbox(
                                name="no_images",
                                label="Number of images",
                                value=str(q.client.no_images) if q.client.no_images else str(1),
                            ),
                            ui.textbox(
                                name="prompt_seed",
                                label="Seed",
                                value=str(q.client.prompt_seed),
                            ),
                            ui.dropdown(
                                name="df_sampling_dropdown",
                                label="Samplers",
                                value="K-LMS",
                                choices=[
                                    ui.choice(name="K-LMS", label="K-LMS(Katherine Crowson)"),
                                    ui.choice(name="DDIM", label="DDIM"),
                                ],
                            ),
                            ui.slider(
                                name="diffusion_n_steps",
                                label="Steps",
                                min=10,
                                max=150,
                                value=q.client.diffusion_n_steps,
                                tooltip="No of steps for image synthesis.",
                            ),
                            ui.slider(
                                name="prompt_guidance_scale",
                                label="Guidance scale",
                                min=3,
                                max=50,
                                step=0.5,
                                value=q.client.prompt_guidance_scale,
                                tooltip="No of steps for image synthesis.",
                            ),
                        ],
                    ),
                ]
            else:
                items = [ui.button(name="prompt_apply", label="Draw")]
                extra_settings = [
                    ui.expander(
                        name="expander",
                        label="Settings",
                        items=[
                            ui.textbox(
                                name="prompt_seed",
                                label="Seed",
                                value=str(q.client.prompt_seed),
                            ),
                            ui.textbox(
                                name="prompt_top_k",
                                label="Top-K",
                                value=str(q.client.prompt_top_k),
                            ),
                            ui.textbox(
                                name="prompt_top_p",
                                label="Top-P",
                                value=str(q.client.prompt_top_p),
                            ),
                            ui.textbox(
                                name="prompt_temp",
                                label="Temperature",
                                value=str(q.client.prompt_temp),
                            ),
                            ui.slider(
                                name="prompt_cond_scale",
                                label="Condition Scale",
                                min=1,
                                max=20,
                                value=q.client.prompt_cond_scale,
                                tooltip="Higher the value, the closer the result is to the prompt (less diversity).",
                            ),
                        ],
                    )
                ]
            q.page["prompt_form"] = ui.form_card(
                ui.box("main", order=1, height="320px", width="980px"),
                items=items
                + [
                    ui.textbox(
                        name="prompt_textbox",
                        label="Prompt",
                        multiline=True,
                        value=q.client.prompt_textbox,
                    )
                ]
                + extra_settings,
            )
            # Build Image grid based on the number of images generated
            display_grid_view(q, q.client.processedimg)
    if save:
        await q.page.save()


async def update_controls(q: Q, save=False):
    q.page["controls"] = get_controls(q)
    if save:
        await q.page.save()


async def update_gif(q: Q):
    del q.page["processed_gif"]
    if q.client.task_choice == "C":
        if q.client.gif_path:
            with open(q.client.gif_path, "rb") as gif_file:
                img_buf = base64.b64encode(gif_file.read()).decode("utf-8")
            q.page["processed_gif"] = await get_gif_generated_card(q, img_buf)
        else:
            q.page["processed_gif"] = get_processed_face_card(
                None,
                title="Generated GIF",
                type="jpg",
                layout_pos="main",
                order=2,
                empty_msg="After applying the edits, click 'Generate GIF' to generate a gif of the transformation!",
            )
        await q.page.save()


async def progress_generate_gif(q: Q):
    q.page["processed_gif"] = get_generate_gif_progress_card(title="")
    await q.page.save()


async def make_base_ui(q: Q, save=False):
    q.app.logo_path = None
    if not q.app.logo_path:
        q.app.logo_path = (await q.run(q.site.upload, ["./img_styler/ui/assets/h2o_logo.svg"]))[0]
    q.page["meta"] = get_meta(q)
    q.page["header"] = get_header(q)
    q.page["user_title"] = get_user_title(q)
    await update_controls(q)
    await update_faces(q)
    await update_processed_face(q)
    await update_gif(q)
    q.page["footer"] = get_footer()
    if save:
        await q.page.save()


def set_username(q: Q):
    first = last = ""
    email = q.auth.username
    names = email.split("@")[0].split(".")
    if len(names) > 1:
        first, *_, last = names
    elif names:
        first = names[0]
        last = ""

    q.user.email = q.auth.username
    q.user.first_name = first.strip().title()
    q.user.last_name = last.strip().title()
    q.user.full_name = f"{q.user.first_name} {q.user.last_name}".strip().title()

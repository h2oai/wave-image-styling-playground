import base64
import os
import random
from h2o_wave import Q

from .components import (
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
from ..utils.dataops import img2buf


async def update_faces(q: Q, save=False):
    if not q.client.source_face or not os.path.exists(q.client.source_face):
        q.client.source_face = random.choice(q.app.source_faces)

    if q.client.task_choice == 'A':
        q.page['source_face'] = get_source_face_card(
            img2buf(q.client.source_face), type='jpg'
        )
    else:
        q.page['source_face'] = get_source_face_card(
            img2buf(q.client.source_face), type='jpg', height='520px', width='500px'
        )
    del q.page['style_face']
    if q.client.task_choice == 'A':
        q.page['style_face'] = get_style_face_card(
            img2buf(q.client.style_face), type='jpg'
        )
    if save:
        await q.page.save()


async def update_processed_face(q: Q, save=False):
    img_buf = img2buf(q.client.processedimg) if q.client.processedimg else None
    del q.page['processed_face']
    if q.client.task_choice == 'A':
        q.page['processed_face'] = get_processed_face_card(img_buf, type='jpg')
    else:
        q.page['processed_face'] = get_processed_face_card(
            img_buf, type='jpg', layout_pos='middle_right', order=2
        )
    if save:
        await q.page.save()


async def update_controls(q: Q, save=False):
    q.page['controls'] = get_controls(q)
    if save:
        await q.page.save()


async def update_gif(q: Q):
    del q.page['processed_gif']
    if q.client.task_choice == 'B':
        if q.client.gif_path:
            with open(q.client.gif_path, "rb") as gif_file:
                img_buf = base64.b64encode(gif_file.read()).decode("utf-8")
            q.page['processed_gif'] = await get_gif_generated_card(q, img_buf)
        else:
            q.page['processed_gif'] = get_processed_face_card(
                None, title="Generated GIF", type='jpg', layout_pos='main',
                order=2,
                empty_msg="After applying the edits, click 'Generate GIF' to generate a gif of the transformation!"
            )
        await q.page.save()


async def progress_generate_gif(q: Q):
    q.page['processed_gif'] = get_generate_gif_progress_card(title="")
    await q.page.save()


async def make_base_ui(q: Q, save=False):
    q.app.logo_path = None
    if not q.app.logo_path:
        q.app.logo_path = (
            await q.run(q.site.upload, ["./img_styler/ui/assets/h2o_logo.svg"])
        )[0]
    q.page['meta'] = get_meta(q)
    q.page['header'] = get_header(q)
    q.page['user_title'] = get_user_title(q)
    await update_controls(q)
    await update_faces(q)
    await update_processed_face(q)
    await update_gif(q)
    q.page['footer'] = get_footer()
    if save:
        await q.page.save()


def set_username(q: Q):
    first = last = ''
    email = q.auth.username
    names = email.split('@')[0].split('.')
    if len(names) > 1:
        first, *_, last = names
    elif names:
        first = names[0]
        last = ''

    q.user.email = q.auth.username
    q.user.first_name = first.strip().title()
    q.user.last_name = last.strip().title()
    q.user.full_name = f'{q.user.first_name} {q.user.last_name}'.strip().title()

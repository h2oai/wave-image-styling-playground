import random
from h2o_wave import Q
from loguru import logger


from .common import make_base_ui, set_username
from ..utils.dataops import get_files_in_dir


async def custom_app_init(q: Q):  # noqa
    q.app.source_faces = get_files_in_dir(dir_path='./images/')
    q.app.style_faces = q.app.source_faces
    logger.debug('Images:')
    for x in q.app.source_faces:
        logger.debug(x)


async def custom_user_init(q: Q):
    q.user.dark_mode = False
    set_username(q)


async def custom_client_init(q: Q):
    if q.app.source_face:
        q.client.source_face = q.app.source_face
    else:
        q.client.source_face = random.choice(q.app.source_faces)
    q.client.style_face = random.choice(q.app.style_faces)
    q.client.task_choice = 'A'
    q.client.z_low = 7
    q.client.z_high = 14
    q.client.diffusion_n_steps = 50
    q.client.prompt_guidance_scale = 7.5
    q.client.prompt_use_source_img = True


async def initialize_app(q: Q):
    # Initialize only once per app instance
    if q.app.initialized:
        return

    logger.info("Initializing App")

    # All logged in users will be added to this during user init
    q.app.users = []

    # Perform all initialization specific to this app
    await custom_app_init(q)

    # Mark the app as initialized
    q.app.initialized = True


async def initialize_user(q: Q):
    user_id = q.auth.subject

    # If this user exists, do nothing
    if user_id in q.app.users:
        return

    logger.info("Initializing User")

    # Save new user
    q.app.users.append(user_id)

    # Perform user initialization specific to this app
    await custom_user_init(q)


async def initialize_client(q: Q):
    if q.client.initialized:
        return

    logger.info("Initializing Client")

    # Perform all initialization specific to this app
    await custom_client_init(q)

    # Crate the first view of the app
    await make_base_ui(q, save=True)

    # Mark the client as initialized
    q.client.initialized = True

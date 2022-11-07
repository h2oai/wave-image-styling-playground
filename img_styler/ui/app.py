from h2o_wave import Q, app, main  # noqa: F401
from loguru import logger

from ..utils.wave_utils import log_args
from . import handlers  # noqa: F401
from .handlers import home, img_capture_save
from .initializers import initialize_app, initialize_client, initialize_user
from .wave_next import handle_on


def on_startup():
    logger.info("Starting App ...")


def on_shutdown():
    logger.info("Stopping App ...")


@app("/", mode="unicast", on_startup=on_startup, on_shutdown=on_shutdown)
async def serve(q: Q):
    logger.info("Serving events...")

    log_args(q.args)

    hash = q.args["#"]
    if hash == "home":
        await home(q)

    # Initialization
    await initialize_app(q)
    await initialize_user(q)
    await initialize_client(q)

    await handle_on(q)

    logger.info("Exit serve")

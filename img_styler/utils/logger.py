import os
import sys
import uuid as uqid
from pathlib import Path

import toml
from loguru import logger as logging


def unique_id():
    # Returns random unique ID.
    return str(uqid.uuid4())


uuid = unique_id()

# Load the config file to read in system settings.
base_path = (Path(__file__).parent / "../configs/").resolve()
app_settings = toml.load(f"{base_path}/settings.toml")

job_dir = app_settings['Scheduler']['JobDir']
job_path = f"{job_dir}/{uuid}"
log_filename = f"{job_path}/styleganwebapp.log"

os.makedirs(os.path.dirname(log_filename), exist_ok=True)

logging = logging.patch(lambda record: record.update(name="StyleGANWebApp"))
logging.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
logging.add(log_filename)

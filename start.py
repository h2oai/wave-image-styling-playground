# This additional installation is in place to help
# resolve cmake error related to dlib.
import errno
import os
import sys
import shlex
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlretrieve

from loguru import logger as logging


def make_dir(path: str):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise Exception("Error reported while creating default directory path.")


base_path = Path(__file__).parent.resolve()
PATH_ON_CLOUD = "/resources/venv/bin/python"
# Check for CLOUD path, if doesn't exist set it to ./venv/bin/python
PYTHON_PATH = "./.venv/bin/python" if os.path.isdir("./.venv/bin/") else PATH_ON_CLOUD
logging = logging.patch(lambda record: record.update(name="ImageStylingArtStudio"))
logging.add(
    sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO"
)

logging.info("Additional installation steps.")
# For tracking jobs
make_dir(f"{base_path}/var/lib/tmp/jobs/output/")
cmd1 = f"{PYTHON_PATH} -m pip install cmake"
subprocess.check_output(shlex.split(cmd1))

cmd2 = f"{PYTHON_PATH} -m pip install dlib==19.24.0"
subprocess.check_output(shlex.split(cmd2))

logging.info("Final Stage: Additional dependencies installed.")

logging.info(f"Download pending model...")
make_dir(f"{base_path}/models/stable_diffusion_v1_4/")
urlretrieve(
    "https://s3.amazonaws.com/ai.h2o.wave-image-styler/public/models/stable_diffusion_v1_4.zip",
    f"{base_path}/models/stable_diffusion_v1_4.zip",
)
shutil.unpack_archive(
    f"{base_path}/models/stable_diffusion_v1_4.zip",
    f"{base_path}/",
)
os.remove(f"{base_path}/models/stable_diffusion_v1_4.zip")

make_dir(f"{base_path}/models/stylegan_nada/")
urlretrieve(
    "https://s3.amazonaws.com/ai.h2o.wave-image-styler/public/models/stylegan_nada.zip",
    f"{base_path}/models/stylegan_nada.zip",
)
shutil.unpack_archive(f"{base_path}/models/stylegan_nada.zip", f"{base_path}/")
os.remove(f"{base_path}/models/stylegan_nada.zip")

# Once all dependencies are installed "Start" AutoInsights.
logging.info("Starting Image Styler.")
DAEMON_PATH = (
    "./.venv/bin/uvicorn"
    if os.path.isdir("./.venv/bin/")
    else "/resources/venv/bin/uvicorn"
)
cmd3 = f"{DAEMON_PATH} img_styler.ui.app:main"
subprocess.check_output(shlex.split(cmd3))
logging.info(f"One should never get here. Something went wrong.")

# This additional installation is in place to help with
# installing special forecast related dependencies which
# involves managing C dependencies. These types of dependencies are difficult
# to manage using `pip install`.
# Below dependencies are needed to resolve cmake error related to dlib.
import subprocess
import shlex
import toml
import os
import errno
from pathlib import Path
import logging


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
PYTHON_PATH = "./venv/bin/python" if os.path.isdir("./venv/bin/") else PATH_ON_CLOUD

logging.info("Additional installation steps.")
# For tracking jobs
make_dir(f"{base_path}/var/lib/tmp/jobs/output/")
cmd1 = f"{PYTHON_PATH} -m pip install cmake==3.22.2"
subprocess.check_output(shlex.split(cmd1))

cmd2 = f"{PYTHON_PATH} -m pip install dlib==19.23.0"
subprocess.check_output(shlex.split(cmd2))
logging.info("Final Stage: Additional dependencies installed.")

# Once all dependencies are installed "Start" AutoInsights.
logging.info("Starting Image Styler.")
DAEMON_PATH = (
    "./venv/bin/uvicorn"
    if os.path.isdir("./venv/bin/")
    else "/resources/venv/bin/uvicorn"
)
cmd3 = f"{DAEMON_PATH} img_styler.ui.app:main"
subprocess.check_output(shlex.split(cmd3))
logging.info(f"One should never get here. Something went wrong.")

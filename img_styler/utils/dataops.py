import base64
import io
import os
from PIL import Image


def img2buf(img_path):
    img = Image.open(img_path)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    img_buf = base64.b64encode(buf.read()).decode('utf-8')
    return img_buf


def buf2img(img, file_name: str):
    # https://stackoverflow.com/questions/2323128/convert-string-in-base64-to-image-and-save-on-filesystem
    _ni = img.split(",")[1]
    image_ = io.BytesIO(base64.b64decode(_ni))
    _img = Image.open(image_)
    img = _img.convert('RGB')
    img.save(file_name, 'JPEG', quality=95)


def remove_file(file_name: str):
    try:
        os.remove(file_name)
    except:
        pass


def get_files_in_dir(dir_path, pattern='', ext=None):
    files = []

    def ext_check(name):
        if ext:
            return name.endswith(f'.{ext}')
        return True

    with os.scandir(dir_path) as it:
        for entry in it:
            if entry.is_file() and ext_check(entry.name) and pattern in entry.name:
                files.append(entry.path)
    return files

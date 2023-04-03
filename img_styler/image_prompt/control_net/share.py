import config
from img_styler.image_prompt.control_net.cldm.hack import disable_verbosity, enable_sliced_attention


disable_verbosity()

if config.save_memory:
    enable_sliced_attention()

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from h2o_wave.core import expando_to_dict
from h2o_wave import Q


style_str = """
<style>
.btn-group button {
  background-color: #04AA6D; /* Green background */
  border: 1px solid green; /* Green border */
  color: white; /* White text */
  padding: 10px 24px; /* Some padding */
  cursor: pointer; /* Pointer/hand icon */
  float: left; /* Float the buttons side by side */
}

/* Add a background color on hover */
.btn-group button:hover {
  background-color: #3e8e41;
}
</style>
"""

html_str = """
<video id="video" width="800" height="600" autoplay></video>
<div style="width:500px;">
    <button class="btn btn-primary" id="snap">Snap</button>
    <button class="btn btn-secondary" id="done">Exit</button>
</div>
<canvas id="canvas" width="800" height="800"></canvas>
"""

js_schema = """
// Grab elements, create settings, etc.
var video = document.getElementById('video');

// Get access to the camera!
if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    // Not adding `{ audio: true }` since we only want video now
    navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
        //video.src = window.URL.createObjectURL(stream);
        //video.play();
        video.srcObject=stream;
        video.play();
    });
}

// Elements for taking the snapshot
var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
var video = document.getElementById('video');

// Trigger photo take
document.getElementById("snap").addEventListener("click", function() {
	context.drawImage(video, 10, 10, 800, 600);
    var myCanvas = document.getElementById('canvas');
    var image = myCanvas.toDataURL("image/png");
    console.log("Image captured", image)
    wave.emit('video', 'click', image);
});
document.getElementById("done").addEventListener("click", function() {
    wave.emit('exit', 'click', 'True');
});
"""


async def capture_img(q: Q):
    events_dict = expando_to_dict(q.events)
    _img = None
    if "the_plot" in events_dict.keys():
        if q.events.the_plot.selected:
            q.page["capture_img"].content = f"You selected {q.events.the_plot.selected}"
    if "video" in events_dict.keys():
        _img = q.events.video.click if q.events.video.click else None
    if "exit" in events_dict.keys():
        q.args.exit_camera = True
    return _img


def draw_boundary(img, x, y, w, h, text: str, image_scale=1):
    _img = Image.fromarray(img)
    draw = ImageDraw.Draw(_img)
    pt_1 = (
        int(x * image_scale),
        int(y * image_scale),
        int((x + w) * image_scale),
        int((y + h) * image_scale),
    )
    draw.rounded_rectangle(pt_1, outline="yellow", width=3, radius=7)

    font = font_manager.FontProperties(family="sans-serif", weight="bold")
    file = font_manager.findfont(font)
    font = ImageFont.truetype(file, 24)
    constant = 5
    draw.text((x + w + constant, y), text, font=font, fill="blue")

    return np.array(_img)

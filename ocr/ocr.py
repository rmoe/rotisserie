import io
import os
import subprocess
import uuid

from sanic import Sanic
from sanic.response import json
import streamlink
import tensorflow as tf
from PIL import Image

app = Sanic()


def load_graph(graph_file):
    """
    Load a frozen TensorFlow graph
    """
    with tf.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    return graph


app.pubg_graph = load_graph("./pubg.pb")
app.fortnite_graph = load_graph("./fortnite.pb")
app.blackout_graph = load_graph("./blackout.pb")
app.ocr_debug = os.environ.get("OCR_DEBUG", False)

# Twitch streams have lots of variations on 720p and some streams
# don't even have that. If we don't specify one of the available formats
# streamlink will fail. It will also fallback to 'best' or 'source' if nothing
# else can be found, though those aren't guaranteed to be valid either.
quality = ("720p", "720", "720p60", "720p60_alt", "best", "source")


def get_stream(url):
    try:
        streams = streamlink.streams(url)
    except Exception as e:
        print(e)
        print(url, "failed")
        return

    for opt in quality:
        if opt in streams:
            return streams[opt]


def get_stream_image(stream_name, crop_dims):
    """Get cropped image from stream.

    :param stream: Name of stream.
    :param crop_dims: Dimensions used to crop image. Format is
        (x, y, w, h) where x and y are the upper left point of the bounding
        rectangle and w and h are width and height in pixels.

    :returns: Cropped PNG image or None
    """
    stream_url = 'https://twitch.tv/{}'.format(stream_name)
    stream = get_stream(stream_url)
    if not stream:
        return

    crop_dims = dict(zip(('x', 'y', 'w', 'h'), crop_dims))
    crop_string = "crop={w}:{h}:{x}:{y}".format(**crop_dims)

    video_url = stream.url
    # Capture the first frame
    pipe = subprocess.run(["ffmpeg", "-i", video_url,
                           "-loglevel", "quiet",
                           "-an",
                           "-f", "image2pipe",
                           "-pix_fmt", "gray",
                           "-vframes", "1",
                           "-filter:v", crop_string,
                           "-vcodec", "png", "-"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    image_data = pipe.stdout
    return image_data


@app.route("/info", methods=["GET"])
async def info(request):
    return json({
        "app": "ocr",
        "version": "0.4",
        "health": "good"
    })


async def _process_image(model, image_data):
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(graph=model, config=config) as sess:
        img_pl = model.get_tensor_by_name("import/input_image_as_bytes:0")
        input_feed = {img_pl: image_data}
        output_feed = [
            model.get_tensor_by_name("import/prediction:0"),
            model.get_tensor_by_name("import/probability:0")
        ]

        res = sess.run(output_feed, input_feed)
        try:
            number = int(res[0])
        except ValueError:
            number = 100

    if app.ocr_debug:
        filename = "debug/{}_{}.png".format(uuid.uuid4(), number)
        with open(filename, 'wb') as f:
            f.write(image_data)

        print("Identified image {} as {} with {:5.2f}% probability.".format(
            filename, number, res[1] * 100))

    return number, res[1]


@app.route("/process_blackout", methods=["POST"])
async def process_blackout(request):
    stream = request.form.get('stream')
    crop_dims = (0, 0, 1280, 720)

    image_data = get_stream_image(stream, crop_dims)
    if not image_data:
        return json({
            "number": 100,
            "prob": 0
        })

    crop_width = 26
    crop_height = 18
    img = Image.open(io.BytesIO(image_data))
    cropped_img = img.crop((1226, 32, 1226 + crop_width, 32 + crop_height))
    output_img = io.BytesIO()
    cropped_img.save(output_img, format='PNG')
    output_img = output_img.getvalue()
    number, prob = await _process_image(app.blackout_graph, output_img)

    # If it says '1' we're probably getting the number of watchers, not the
    # number of remaining players.
    if number >= 1 and number <= 6:
        print("trying again")
        cropped = img.crop((1170, 32, 1170 + crop_width, 32 + crop_height))
        output_img = io.BytesIO()
        cropped.save(output_img, format='PNG')
        output_img = output_img.getvalue()
        number, prob = await _process_image(app.blackout_graph, output_img)

    return json({
        "number": number or 100,
        "probability": prob
    })


@app.route("/process_fortnite", methods=["POST"])
async def process_fortnite(request):
    stream = request.form.get('stream')

    crop_dims = (1188, 204, 22, 18)
    image_data = get_stream_image(stream, crop_dims)
    if not image_data:
        return json({
            "number": 100,
            "prob": 0
        })

    number, prob = await _process_image(app.fortnite_graph, image_data)

    return json({
        "number": number or 100,
        "prob": prob
    })


@app.route("/process_pubg", methods=["POST"])
async def process_pubg(request):
    stream = request.form.get('stream')

    crop_dims = (1191, 22, 23, 21)
    image_data = get_stream_image(stream, crop_dims)
    if not image_data:
        return json({
            "number": 100,
            "prob": 0
        })

    im = Image.open(io.BytesIO(image_data)).convert('L')
    px = im.load()

    left = px[15, 9]
    center = px[16, 9]
    right = px[17, 9]

    # Before a game has started the top right looks like:
    # XX | Joined
    # Where XX is the number of players.
    #
    # The area we crop assumes it looks like:
    # XX | Alive
    #
    # Because 'joined' has one more character than 'alive' it causes
    # the leftmost number to be shifted outside of our capture area.
    # When this happens the model correctly identifies the only number
    # it sees causing that stream to be erroneously marked as the one
    # with the fewest players. E.g. 96 | Joined will be cropped into
    # 6 | which will be identified as 6 instead of 96.
    #
    # To remedy this we can check for the prescence of a vertical line
    # and return 100 for those streams. If a bar is present then the game
    # has not started and we can safely assume this stream does not have
    # the fewest remaining players.
    #
    # values are low (dark) to high (light) so a high, low, high sequence
    # of pixels is what a line looks like
    #
    # if left and right are within 10%
    # AND
    # if center is 25% less than left and right
    # that looks like a line
    if 0.90 * right < left < 1.10 * right:
        if center < 0.75 * right:
            if app.ocr_debug:
                print("Skipping possible 'joined' image.")

            return json({
                "number": 100,
                "prob": 0
            })

    number, prob = await _process_image(app.pubg_graph, image_data)

    return json({
        "number": number,
        "prob": prob
    })


if __name__ == "__main__":
    cpus = len(os.sched_getaffinity(0))
    app.run(host="0.0.0.0", port=3001, workers=cpus)

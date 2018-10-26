import io
import os
import uuid
import streamlink
import subprocess

from sanic import Sanic
from sanic.response import json
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


#app.pubg_graph = load_graph("./pubg.pb")
#app.fortnite_graph = load_graph("./fortnite.pb")
app.blackout_graph = load_graph("./blackout.pb")
app.ocr_debug = os.environ.get("OCR_DEBUG", False)
quality = ("720p", "720", "720p60", "720p60_alt", "best", "source")


@app.route("/info", methods=["GET"])
async def info(request):
    return json({
        "app": "ocr",
        "version": "0.3",
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

    return number


def get_stream(url):
    streams = streamlink.streams(url)

    for opt in quality:
        if opt in streams:
            return streams[opt]


@app.route("/process_blackout", methods=["POST"])
async def process_blackout(request):

    stream_url = "http://twitch.tv/{}".format(request.form.get("stream"))
    stream = get_stream(stream_url)

    if not stream:
        return json({"number": 100})

    video_url = stream.url

    pipe = subprocess.run(["ffmpeg", "-i", video_url,
        "-loglevel", "quiet",
        "-an",
        "-f", "image2pipe",
        "-pix_fmt", "gray",
        "-vframes", "1",
        "-filter:v", "crop=26:18:1226:32",
        "-vcodec", "png", "-"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    image_data = pipe.stdout
    number = await _process_image(app.blackout_graph, image_data)

    return json({
        "number": number or 100
    })


@app.route("/process_fortnite", methods=["POST"])
async def process_fortnite(request):
    if not request.files:
        return json({
            "number": 100
        })

    image_data = request.files.get("image").body
    number = await _process_image(app.fortnite_graph, image_data)

    return json({
        "number": number or 100
    })


@app.route("/process_pubg", methods=["POST"])
async def process_pubg(request):
    if not request.files:
        return json({
            "number": 100
        })

    image_data = request.files.get("image").body

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
                "number": 100
            })

    number = await _process_image(app.pubg_graph, image_data)

    return json({
        "number": number
    })


if __name__ == "__main__":
    cpus = len(os.sched_getaffinity(0))
    app.run(host="0.0.0.0", port=3001, workers=cpus)

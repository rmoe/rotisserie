import base64
import flask
import os
import tensorflow as tf
import streamlink
import subprocess
import logging

app = flask.Flask(__name__)
graph = None
logging.basicConfig(format='%(message)s')

# Twitch streams have lots of variations on 720p and some streams
# don't even have that. If we don't specify one of the available formats
# streamlink will fail. It will also fallback to 'best' or 'source' if nothing
# else can be found, though those aren't guaranteed to be valid either.
quality = ("720p", "720", "720p60", "720p60_alt", "best", "source")


def load_graph():
    with tf.gfile.GFile("fortnite.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    return graph


def get_stream(url):
    streams = streamlink.streams(url)
    for opt in quality:
        if opt in streams:
            return streams[opt]


def parse_stream(stream_name, token):
    stream = get_stream('twitch.tv/{}'.format(stream_name))
    url = stream.url

    # Capture the first frame
    pipe = subprocess.run(["ffmpeg", "-i", url,
                           "-loglevel", "quiet",
                           "-an",
                           "-f", "image2pipe",
                           "-pix_fmt", "gray",
                           "-vframes", "1",
                           "-filter:v", "crop=22:17:1184:212",
                           "-vcodec", "png", "-"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    return pipe.stdout


@app.route("/init", methods=["POST"])
def init():
    global graph
    if graph is None:
        graph = load_graph()

    return ('OK', 200)


@app.route("/run", methods=["POST"])
def run():

    req = flask.request.get_json(force=True, silent=True)
    args = req["value"]
    logging.warn(req)
    stream = args.get("stream")
    token = args.get("token")
    logging.warn(stream)
    logging.warn(token)

    image_data = parse_stream(stream, token)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(graph=graph, config=config) as sess:
        img_pl = graph.get_tensor_by_name("import/input_image_as_bytes:0")
        input_feed = {img_pl: image_data}
        output_feed = [
            graph.get_tensor_by_name("import/prediction:0"),
            graph.get_tensor_by_name("import/probability:0")
        ]

        res = sess.run(output_feed, input_feed)
        try:
            number = int(res[0])
        except ValueError:
            number = 100

    resp = {
        'number': number,
        'probability': res[1],
        'status_code': 200
    }

    if args.get("debug"):
        resp['image_data'] = base64.b64encode(image_data).decode("utf-8")

    return flask.jsonify(resp)


if __name__ == "__main__":
    port = int(os.getenv('FLASK_PROXY_PORT', 8080))
    app.run(host='0.0.0.0', port=port)

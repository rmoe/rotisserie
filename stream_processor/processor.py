import json
import redis
import requests
import os
import streamlink
import subprocess
import time


db = redis.StrictRedis(host='redis-master',
                       password=os.getenv('REDIS_PASSWORD'))

# This is a redis set that contains a list of stream names
read_key = os.getenv('REDIS_READ_KEY', 'stream-list')

# Results are written to a sorted set. Sorted sets contain a string
# and a score and are sorted in ascending order based on the score.
# This makes it so the list of streams is always sorted by remaining players.
write_key = os.getenv('REDIS_WRITE_KEY', 'stream-by-alive')

# Twitch OAUTH token
token = os.getenv("TOKEN")

# Twitch streams have lots of variations on 720p and some streams
# don't even have that. If we don't specify one of the available formats
# streamlink will fail. It will also fallback to 'best' or 'source' if nothing
# else can be found, though those aren't guaranteed to be valid either.
quality = ("720", "720p", "720p60", "720p60_alt", "best", "source")


def main():
    """Main loop

    Loops and pops values from the set of streams. For each of those streams
    it captures 4 seconds of the video, captures a screenshot of the first
    frame and submits it to the OCR service to get a value. That value, along
    with the stream name is added to the sorted set of parsed streams.
    """
    while True:
        stream_name = db.spop(read_key)

        if not stream_name:
            time.sleep(3)
            continue

        stream_name = stream_name.decode('utf-8')

        if not stream_name:
            continue

        stream = get_stream('twitch.tv/{}'.format(stream_name))
        if not stream:
            continue

        screenshot = take_screenshot(stream.url)
        if not screenshot:
            continue

        value = ocr(screenshot)

        db.zadd(write_key, value, stream_name)


def take_screenshot(stream_url):
    pipe = subprocess.run(['ffmpeg', "-i", stream_url,
                           "-loglevel", "quiet",
                           "-an",
                           "-f", "image2pipe",
                           "-pix_fmt", "gray",
                           "-vframes", "1",
                           "-filter:v", "crop=22:22:1190:20",
                           "-vcodec", "png", "-"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    return pipe.stdout


def ocr(file_data):
    files = {'image': file_data}
    req = requests.post("http://rotisserie-ocr:3001/process_pubg", files=files)

    data = json.loads(req.text)
    return data['number']


def get_stream(url):
    streams = streamlink.streams(url)
    for opt in quality:
        if opt in streams:
            return streams[opt]


if __name__ == "__main__":
    main()

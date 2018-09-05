FROM tensorflow/tensorflow:1.7.1-py3

WORKDIR /

RUN echo "force-unsafe-io" > /etc/dpkg/dpkg.cfg.d/02apt-speedup && \
    echo "Acquire::http {No-Cache=True;};" > /etc/apt/apt.conf.d/no-cache && \
    pip install flask streamlink psutil && \
    apt-get update && apt-get -y install --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY fortnite.pb action.py /

CMD ["python", "-u", "action.py"]

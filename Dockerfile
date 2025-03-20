# File name: Dockerfile
FROM rayproject/ray-ml:2.9.2-py310
WORKDIR /serve_app
COPY requirements.txt .
RUN pip install -r requirements.txt

USER root
# RUN chmod 777 /serve_app/audio_files

ENV TZ=Asia/Singapore


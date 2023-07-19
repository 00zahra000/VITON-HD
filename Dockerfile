# pull official base image
# FROM python:3.8
FROM tiangolo/uvicorn-gunicorn:python3.8
WORKDIR /app/
COPY req.txt /app/
RUN pip install -r req.txt
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY . /app/
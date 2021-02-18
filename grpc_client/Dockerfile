FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

RUN apt-get update && apt-get install -y libgl1-mesa-glx ffmpeg libsm6 libxext6 ### qt5-default
RUN pip install --upgrade pip grpcio-tools==1.32.0 tensorflow-serving-client opencv-python
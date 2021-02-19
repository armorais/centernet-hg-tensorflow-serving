FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

# base dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx ffmpeg libsm6 libxext6 protobuf-compiler && \
    pip install --upgrade pip grpcio-tools==1.32.0 tensorflow-serving-client pyyaml

# tensorflow object detection API
RUN git clone --depth 1 https://github.com/tensorflow/models && cd models/research/ && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && python3 -m pip install -q . && \
    pip install --upgrade opencv-python==4.4.0.46
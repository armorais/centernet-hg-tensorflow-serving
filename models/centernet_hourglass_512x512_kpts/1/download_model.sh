#!/bin/bash
wget -nc http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz
tar -xvf centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz
mv centernet_hg104_512x512_kpts_coco17_tpu-32/saved_model/* .
rm centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz
rm -r centernet_hg104_512x512_kpts_coco17_tpu-32
import logging
import tensorflow as tf
import numpy as np
import grpc
from tensorflow_serving_client.protos import prediction_service_pb2_grpc, predict_pb2
from tensorflow_serving_client.proto_util import copy_message

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

class TensorflowServingClient(object):

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.channel = grpc.insecure_channel('%s:%s' % (host, port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)

    def execute(self, request, timeout=10.0):
        return self.stub.Predict(request, timeout)

    def make_prediction(self, input_data, input_tensor_name, timeout=10.0, model_name=None):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name or 'model'

        copy_message(tf.make_tensor_proto(input_data.astype(dtype=np.uint8), dtype='uint8'), request.inputs[input_tensor_name])
        response = self.execute(request, timeout=timeout)

        results = {}
        for key in response.outputs:
            tensor_proto = response.outputs[key]
            nd_array = tf.make_ndarray(tensor_proto)
            results[key] = nd_array

        return results
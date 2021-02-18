import os, sys, logging, time
import numpy as np
import cv2
from centernet_grpc_client import TensorflowServingClient

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

time.sleep(15)

# Get environment variables
SERVER_HOST = os.getenv('SERVER_HOST')
SERVER_PORT = os.getenv('SERVER_PORT')
MODEL_NAME = os.getenv('MODEL_NAME')

logging.info(f'Server host: {SERVER_HOST}')
logging.info(f'Server port: {SERVER_PORT}')

try:
    grpc_client = TensorflowServingClient(SERVER_HOST, SERVER_PORT)
except Exception as e:
  logging.error("Exception occurred while creating tf serving client", exc_info=True)
  sys.exit(1)
try:
    cap = cv2.VideoCapture('/samples/test.mp4')
except Exception as e:
  logging.error("Exception occurred while opening the video file", exc_info=True)
  sys.exit(1)
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #serving_default_input_tensor
    results = grpc_client.make_prediction(np.expand_dims(frame, axis=0),"input_tensor",timeout=10,model_name=MODEL_NAME)
    logging.info(results)
    # TODO: add results parsing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
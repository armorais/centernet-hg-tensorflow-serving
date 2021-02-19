import os, sys, time, logging
import numpy as np
import cv2
import yaml
from centernet_grpc_client import TensorflowServingClient
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)

time.sleep(15)

# Get environment variables
SERVER_HOST = os.getenv('SERVER_HOST')
SERVER_PORT = os.getenv('SERVER_PORT')
MODEL_NAME = os.getenv('MODEL_NAME')

logging.info(f'Server host: {SERVER_HOST}')
logging.info(f'Server port: {SERVER_PORT}')

def draw_dets(image, results, category_index, coco_keypoints):
  label_id_offset = 0
  # Use keypoints if available in detections
  keypoints, keypoint_scores = None, None
  if 'detection_keypoints' in results:
    keypoints = results['detection_keypoints'][0]
    keypoint_scores = results['detection_keypoint_scores'][0]
  viz_utils.visualize_boxes_and_labels_on_image_array(
        image,
        results['detection_boxes'][0],
        (results['detection_classes'][0] + label_id_offset).astype(int),
        results['detection_scores'][0],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=coco_keypoints)

def detect(cfg):
  try:
      grpc_client = TensorflowServingClient(SERVER_HOST, SERVER_PORT)
  except Exception as e:
    logging.error("Exception occurred while creating tf serving client", exc_info=True)
    sys.exit(1)
  try:
      cap = cv2.VideoCapture(cfg['source'])
  except Exception as e:
    logging.error(f"Exception occurred while opening the video file: {cfg['source']}", exc_info=True)
    sys.exit(1)

  coco_labels = cfg['coco_labels']
  category_index = label_map_util.create_category_index_from_labelmap(coco_labels, use_display_name=True)
  coco_keypoints = cfg['coco17_human_pose_keypoints']

  while(cap.isOpened()):
      ret, frame = cap.read()
      results = grpc_client.make_prediction(np.expand_dims(frame, axis=0),"input_tensor",timeout=10,model_name=MODEL_NAME)
      image_np_with_detections = frame.copy()
      draw_dets(image_np_with_detections, results, category_index, coco_keypoints)
      cv2.imshow("output", image_np_with_detections)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
    with open('config/config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    detect(cfg)
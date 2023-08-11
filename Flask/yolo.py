import numpy as np
import onnxruntime
import os
import cv2
from PIL import Image

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def pred(img_path , conf_thresold = 0.3 , iou_threshold = 0.3):
  '''
  img_path : single image path -> str
  conf_thresold : Confidence value
  iou_threshold : IoU of boxes
  '''

  #The code you have provided sets up an onnxruntime.SessionOptions object
  opt_session = onnxruntime.SessionOptions()
  opt_session.enable_mem_pattern = False
  opt_session.enable_cpu_mem_arena = False
  opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

  #Set model path
  model_path = os.path.join(os.path.dirname(__file__) , 'best.onnx')
  EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

  ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list)

  # find input
  model_inputs = ort_session.get_inputs()
  input_names = [model_inputs[i].name for i in range(len(model_inputs))]
  input_shape = model_inputs[0].shape

  # input image
  image = cv2.imread(img_path)
  image_height, image_width = image.shape[:2]
  input_height, input_width = input_shape[2:]
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  resized = cv2.resize(image_rgb, (input_width, input_height)) # Resize to 640*640

  # Scale input pixel value to 0 to 1
  input_image = resized / 255.0
  input_image = input_image.transpose(2,0,1)
  input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
  
  # output name 
  model_output = ort_session.get_outputs()
  output_names = [model_output[i].name for i in range(len(model_output))]

  #Infreance
  outputs = ort_session.run(output_names, {input_names[0]: input_tensor})[0] # input tensor 1,3,640,640 ---> 1, 124,8400

  # squeeze = 1, 124,8400 ---> 124,8400
  # .T --> 8400,124
  predictions = np.squeeze(outputs).T

  # Filter out object confidence scores below threshold
  scores = np.max(predictions[:, 4:], axis=1)
  predictions = predictions[scores > conf_thresold, :]
  scores = scores[scores > conf_thresold]

  #find class idx
  class_ids = np.argmax(predictions[:, 4:], axis=1)

  # Get bounding boxes for each object
  boxes = predictions[:, :4]

  #rescale box
  input_shape = np.array([input_width, input_height, input_width, input_height])
  boxes = np.divide(boxes, input_shape, dtype=np.float32)
  boxes *= np.array([image_width, image_height, image_width, image_height])
  boxes = boxes.astype(np.int32)
  boxes

  indices = nms(boxes, scores, iou_threshold)

  class_names = {0: 'Afghan_hound', 1: 'African_hunting_dog', 2: 'Airedale', 3: 'American_Staffordshire_terrier', 4: 'Appenzeller', 5: 'Australian_terrier',
                 6: 'Bedlington_terrier', 7: 'Bernese_mountain_dog', 8: 'Blenheim_spaniel', 9: 'Border_collie', 10: 'Border_terrier',
                 11: 'Boston_bull', 12: 'Bouvier_des_Flandres', 13: 'Brabancon_griffon', 14: 'Brittany_spaniel', 15: 'Cardigan',
                 16: 'Chesapeake_Bay_retriever', 17: 'Chihuahua', 18: 'Dandie_Dinmont', 19: 'Doberman', 20: 'English_foxhound',
                 21: 'English_setter', 22: 'English_springer', 23: 'EntleBucher', 24: 'Eskimo_dog', 25: 'French_bulldog',
                 26: 'German_shepherd', 27: 'German_short-haired_pointer', 28: 'Gordon_setter', 29: 'Great_Dane', 30: 'Great_Pyrenees',
                 31: 'Greater_Swiss_Mountain_dog', 32: 'Ibizan_hound', 33: 'Irish_setter', 34: 'Irish_terrier', 35: 'Irish_water_spaniel',
                 36: 'Irish_wolfhound', 37: 'Italian_greyhound', 38: 'Japanese_spaniel', 39: 'Kerry_blue_terrier', 40: 'Labrador_retriever',
                 41: 'Lakeland_terrier', 42: 'Leonberg', 43: 'Lhasa', 44: 'Maltese_dog', 45: 'Mexican_hairless',
                 46: 'Newfoundland', 47: 'Norfolk_terrier', 48: 'Norwegian_elkhound', 49: 'Norwich_terrier', 50: 'Old_English_sheepdog',
                 51: 'Pekinese', 52: 'Pembroke', 53: 'Pomeranian', 54: 'Rhodesian_ridgeback', 55: 'Rottweiler',
                 56: 'Saint_Bernard', 57: 'Saluki', 58: 'Samoyed', 59: 'Scotch_terrier', 60: 'Scottish_deerhound',
                 61: 'Sealyham_terrier', 62: 'Shetland_sheepdog', 63: 'Shih-Tzu', 64: 'Siberian_husky', 65: 'Staffordshire_bullterrier',
                 66: 'Sussex_spaniel', 67: 'Tibetan_mastiff', 68: 'Tibetan_terrier', 69: 'Walker_hound', 70: 'Weimaraner',
                 71: 'Welsh_springer_spaniel', 72: 'West_Highland_white_terrier', 73: 'Yorkshire_terrier', 74: 'affenpinscher', 75: 'basenji',
                 76: 'basset', 77: 'beagle', 78: 'black-and-tan_coonhound', 79: 'bloodhound', 80: 'bluetick',
                 81: 'borzoi', 82: 'boxer', 83: 'briard', 84: 'bull_mastiff', 85: 'cairn',
                 86: 'chow', 87: 'clumber', 88: 'cocker_spaniel', 89: 'collie', 90: 'curly-coated_retriever',
                 91: 'dhole', 92: 'dingo', 93: 'flat-coated_retriever', 94: 'giant_schnauzer', 95: 'golden_retriever',
                 96: 'groenendael', 97: 'keeshond', 98: 'kelpie', 99: 'komondor', 100: 'kuvasz',
                 101: 'malamute', 102: 'malinois', 103: 'miniature_pinscher', 104: 'miniature_poodle', 105: 'miniature_schnauzer',
                 106: 'otterhound', 107: 'papillon', 108: 'pug', 109: 'redbone', 110: 'schipperke',
                 111: 'silky_terrier', 112: 'soft-coated_wheaten_terrier', 113: 'standard_poodle', 114: 'standard_schnauzer', 115: 'toy_poodle',
                 116: 'toy_terrier', 117: 'vizsla', 118: 'whippet', 119: 'wire-haired_fox_terrier'}


  #Draw image
  image_draw = image.copy()
  for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
      print(f"bbox : {bbox} , score: {score} , label : {label}")
      bbox = bbox.round().astype(np.int32).tolist()
      cls_id = int(label)
      cls = class_names[cls_id]
      color = (0,255,0)
      cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
      cv2.putText(image_draw,
                  f'{cls}:{int(score*100)}', (bbox[0], bbox[1] - 2),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.60, [225, 255, 255],
                  thickness=1)
      
  return Image.fromarray(cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB))


def process_image(filename):
    # # Load the image using PIL or any other library
    # img = Image.open(filename)

    # Image processing 
    processed_img = pred(filename , conf_thresold = 0.3 , iou_threshold = 0.3)  # Just an example rotation
    
    processed_img.save(filename)
    
    return filename

#pip install onnxruntime

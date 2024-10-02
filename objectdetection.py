import torch
import cv2
import json
import numpy as np
from ultralytics import YOLO
import argparse

# Using set 2 for 1m and set 3 for 5m images

# Define the parser
parser = argparse.ArgumentParser(description='object detection script')
# Declare an argument (`--xxxx`), saying that the 
# corresponding value should be stored in the `xxxx` 
# field, and using a default value if the argument 
# isn't given
parser.add_argument('--altitude', action="store", dest='altitude', default=1)
parser.add_argument('--index', action="store", dest='index', default=1)

# Now, parse the command line arguments and store the 
# values in the `args` variable
args = parser.parse_args()

# Individual arguments can be accessed as attributes...
# print args.xxxx
file_index = str(args.index)
altitude = str(args.altitude)

# Load the YOLOv9 model (can be replaced this with a custom model if trained)
model = YOLO('C:/Users/f_loe/Projects/Yolo/yolov9-gpu/yolov9c.pt')  # Path to YOLOv9 model weights
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov9c.pt', source='local')

########### USE ONLY FOR YOLOV8 ##################
# # Load the image - 
# image_path = 'C:/Users/f_loe/Projects/Yolo/yolov9-gpu/data/images/DJI/1m/dji_1m_1.JPG'  # Replace with your image file path
# image = cv2.imread(image_path)

# # Perform object detection
# results = model(image)
########### USE ONLY FOR YOLOV8 ##################

############ YOLOV9 ############
# Run inference with a confidence threshold of 0.01
results = model(f'C:/Users/f_loe/Projects/Yolo/yolov9-gpu/data/images/DJI/set_3/{altitude}m/dji_{altitude}m_{file_index}.JPG', conf=0.005) # Replace with your image file path
############ YOLOV9 ############

# result_img = results[0].
# cv2.imwrite('output_image.jpg', result_img)

# Extract detection results (bounding boxes, class names, and confidence scores)
detections = []
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0])  # Class id
        cls_name = result.names[cls]  # Class name
        
        # Append to the list of detections
        detections.append({
            'class_name': cls_name,
            'class_id': cls,
            'confidence': conf,
            'bounding_box': {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            }
        })
    # Save the result image along with the JSON result file
    image_file = f'detection_results_dji_{altitude}m_{file_index}.jpg'
    result.save(image_file)



# Save detection results to a JSON file
output_file = f'detection_results_dji_{altitude}m_{file_index}.json'
with open(output_file, 'w') as f:
    json.dump(detections, f, indent=4)

print(f"Detected {len(detections)} objects in image.")
print(f"Detection results JSON file saved to {output_file}")
print(f"Detection results image file saved to {image_file}")

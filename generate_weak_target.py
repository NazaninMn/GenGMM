import os
import json
from PIL import Image
import numpy as np
import random 
import cv2

with open('/data/target_samples.json', 'r') as json_file:
    unlabeled_samples = json.load(json_file)
    
    
for img in unlabeled_samples:
    dir_ = os.path.join('/data/cityscapes/gtFine/train/',img['ann']['seg_map'])
    if 'zurich' in os.path.join('/data/cityscapes/gtFine/weak_train/',img['ann']['seg_map']):
        img_ = np.array(Image.open(dir_))
        unique_classes = np.unique(img_)
        mask_nahayee = np.ones(img_.shape)*255
        for label in unique_classes:
            if label != 255:
                # Create a Pillow ImageDraw object
                mask_label = img_==label
    #             print(label, mask_label.sum())

                one_coordinates = [(row, col) for row in range(len(mask_label)) for col in range(len(mask_label[0])) if
                                   mask_label[row][col] == 1]

                circle_center = random.choice(one_coordinates)

                circle_radius = 4
                    
                circle_color = int(label)  # Red color (R, G, B)

                mask_nahayee[circle_center[0],circle_center[1]]=int(label)
                mask = np.ones_like(img_)*-1
                cv2.circle(mask, (circle_center[1],circle_center[0]), circle_radius, circle_color,thickness=-1)

                mask_nahayee[(mask>-0.1)*(mask_label>0)]=int(label)

        image = Image.fromarray(mask_nahayee.astype(np.uint8), mode='L')

                # Save the image as a PNG file
        image.save(os.path.join('/data/cityscapes/gtFine/weak_train/',img['ann']['seg_map']))





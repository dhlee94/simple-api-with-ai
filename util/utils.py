
import numpy as np
import cv2
import io
from io import BytesIO
import os
import sys
import base64
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.model import EfficientNet
import torch

def load_model(in_channels: int=1, out_channels: int=1, pre_train: bool=False, 
               model_path: str=None, device: torch.device=torch.device('cuda')):
    model = EfficientNet(in_channels=1, out_channels=1, pre_train=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    return model
    
def transform_image(image_bytestream, interpolation=cv2.INTER_AREA):
    image = cv2.imdecode(np.fromstring(image_bytestream, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image.shape[0]!=420 or image.shape[1] != 420:
         image = cv2.resize(np.array(image), (420, 420), interpolation=interpolation)
    return image

def normalization(input):
    min_value = torch.min(input.reshape(input.shape[0], -1))
    max_value = torch.max(input.reshape(input.shape[0], -1))
    return (input-min_value)/(max_value - min_value)

def from_image_to_bytes(img):
    encoded = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
    return encoded

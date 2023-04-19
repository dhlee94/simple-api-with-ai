import requests
import os
import datetime
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import json
import base64

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required=False, default=None, type=str, help='파일의 경로')
    parser.add_argument('--type', required=False, default='Classification', type=str, help='GradCAM or Classification')
    args = parser.parse_args()
    return args.filename, args.type

if __name__=='__main__':
    FileName, Type = get_arguments()
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),FileName)
    now1 = datetime.datetime.now()
    if Type=='Classification':
        resp = requests.post("http://127.0.0.1:8080/predict",
                            files={"file":open(path, 'rb')}).json()
        print(f'Model Output Result : {resp}')
    elif Type=='GradCAM':
        resp = requests.post("http://127.0.0.1:8080/gradcam",
                            files={"file":open(path, 'rb')}).json()
        now2 = datetime.datetime.now()
        result = base64.b64decode(resp)
        jpg = np.frombuffer(result, dtype=np.uint8)
        img = cv2.imdecode(jpg, cv2.IMREAD_GRAYSCALE)
    else:
        print(f'{Type} is not filled out correctly ')
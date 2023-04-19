from core.function import get_prediction, get_gradcam
import json
import datetime
import torch
import io
from torch.backends import cudnn
from typing import Optional
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from util.utils import from_image_to_bytes

cudnn.benchmark = False
cudnn.deterministic = True
app = FastAPI()

@app.post('/predict')
async def predict(file: UploadFile):
    img_bytestream = await file.read()
    output = (get_prediction(image_bytestream=img_bytestream).squeeze(0)>0.5)
    return json.dumps(output, indent=4, ensure_ascii=False)

@app.post('/gradcam')
async def gradcam(file: UploadFile):
    img_bytestream = await file.read()
    output = get_gradcam(image_bytestream=img_bytestream).squeeze(0)*255
    output = from_image_to_bytes(output)
    return json.dumps(output)

if __name__=="__main__":
    pass
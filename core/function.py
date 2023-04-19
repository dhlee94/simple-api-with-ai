import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from util.utils import load_model, transform_image, normalization
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

cuda = torch.cuda.is_available()
device = torch.device('cuda') if cuda else torch.device('cpu')
model=load_model(in_channels=1, out_channels=1, pre_train=False, 
            model_path='./model/weight/model.pth', device=device)
target_layers = [model.layer.features[-1][0]]
targets = [ClassifierOutputTarget(0)]
model.eval()
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True if cuda else False)

def get_prediction(image_bytestream):
    return inference(image_bytestream=image_bytestream, model=model, device=device)

def get_gradcam(image_bytestream):
    return grad_cam(image_bytestream=image_bytestream, cam=cam, targets=targets, device=device)

def inference(image_bytestream, model, device):
    with torch.no_grad():
        input = transform_image(image_bytestream=image_bytestream)
        input = torch.Tensor(input)
        input = normalization(input)
        input = input.to(device)
    return torch.sigmoid(model(input.unsqueeze(0).unsqueeze(0)))

def grad_cam(image_bytestream, cam, targets, device):
    input = transform_image(image_bytestream=image_bytestream)
    input = torch.Tensor(input)
    input = normalization(input).unsqueeze(0).unsqueeze(0)
    input = input.to(device)
    output = cam(input_tensor=input, targets=targets)
    return output

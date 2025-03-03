# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 11:35:00 2025

@author: SQamar
"""

import torch
from model import ScaleFusionNet
from data_preparation import ISICDataset
import cv2

def predict(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ScaleFusionNet().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    image = cv2.imread(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(image)
    
    pred = (pred > 0.5).float().squeeze().cpu().numpy()
    return pred
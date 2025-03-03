# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 11:34:14 2025

@author: SQamar
"""

import torch
from model import ScaleFusionNet
from data_preparation import get_loaders
from losses import DiceBCELoss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ScaleFusionNet().to(device)
    
    train_loader, val_loader = get_loaders()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = DiceBCELoss()
    
    for epoch in range(200):
        model.train()
        for images, masks in train_loader:
            outputs = model(images.to(device))
            loss = criterion(outputs, masks.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for images, masks in val_loader:
                outputs = model(images.to(device))
                val_loss += criterion(outputs, masks.to(device))
            
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

if __name__ == "__main__":
    main()
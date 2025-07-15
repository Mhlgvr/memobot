import torch
import torch.nn as nn
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_model():
    model = models.resnet18()
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load('data/best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    model = init_model()

import torch
import torch.nn as nn
from torchvision import models

device = torch.device('cpu')

model = models.resnet18()
model.fc = nn.Linear(512, 2)
model.load_state_dict(torch.load('data/dataset/best_model.pth', map_location=device))



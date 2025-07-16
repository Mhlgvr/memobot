import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from inference import BaseModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResnetModel(BaseModel):
    def __init__(self, model_path=None):
        self.model = self.load(model_path)


    
    def load(self, model_path):
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 2)

        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=device))

        model.to(device)
        model.eval()
        return model
    
    def preprocess(self, img):
        transfrom = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img = transform(img)
        return img.to(device).unsqueeze(0)

    def predict(self, img):
        img = self.preprocess(img)
        with torch.no_grad():
            output = self.model(img)
        return output.softmax(dim=1).argmax(dim=1)


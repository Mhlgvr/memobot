from torchvision import models
from inference import BaseModel
import torch.nn as nn
import torch



class EfficientNet(BaseModel):
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.model = self.load(model_path)

    def load(self, model_path):
        model = models.efficientnet_b0
        model._fc = nn.Linear(model._fc.in_features, 2)
        model.to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def preprocess(self, img_path):
        return super().preprocess(img_path)
    
    def predict(self, img):
        img = self.preprocess(img)
        img = img.to(device).unsqueeze(0)
        with torch.no_grad():
            output = self.model(img)
        return output.sofmax(dim=1).argmax(dim=1)
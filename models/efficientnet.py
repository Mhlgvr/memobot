from torchvision import models
import torch.nn as nn
import torch
from .inference import BaseModel
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
project_dir = os.path.dirname(os.getcwd())


classifier_path = os.path.join(project_dir, 'data/classification/efficientnet.pth')
regressor_path = os.path.join(project_dir, 'data/regression/efficientnet.pth')


class EfficientNetMemeClassifier(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = self.load()

    def load(self):
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(1280, 2)
        model.load_state_dict(torch.load(classifier_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    @BaseModel.predict_wrapper
    def predict(self, output):
        probs = output.softmax(dim=1)
        predicted = probs.argmax(dim=1).item()
        return predicted, probs.cpu().numpy()


class EfficientNetMemeRegressor(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = self.load()

    def load(self):
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(1280, 3)
        model.load_state_dict(torch.load(regressor_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    @BaseModel.predict_wrapper
    def predict(self, output):
        return output[0][0].item(), output[0][1].item(), output[0][2].item()

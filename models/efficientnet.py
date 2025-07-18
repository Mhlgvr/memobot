from torchvision import models
import torch.nn as nn
import torch
from .inference import BaseModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classifier_path = '/Users/mhlgvr/Documents/Yandex.Disk.localized/CU2/AI/Bootcamps 2025/memobot/data/classification/efficientnet.pth'
regressor_path = '/Users/mhlgvr/Documents/Yandex.Disk.localized/CU2/AI/Bootcamps 2025/memobot/data/regression/efficientnet.pth'

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

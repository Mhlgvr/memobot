import torch
import torch.nn as nn
from torchvision import models
from .inference import BaseModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classifier_path = '/Users/mhlgvr/Documents/Yandex.Disk.localized/CU2/AI/Bootcamps 2025/memobot/data/classification/resnet50.pth'
regressor_path = '/Users/mhlgvr/Documents/Yandex.Disk.localized/CU2/AI/Bootcamps 2025/memobot/data/regression/resnet50.pth'


class ResNetMemeClassifier(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = self.load()

    def load(self, classifier_path=classifier_path):
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 2)

        model.load_state_dict(torch.load(classifier_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    

    @BaseModel.predict_wrapper
    def predict(self, output):
        probs = output.softmax(dim=1)
        predicted = probs.argmax(dim=1).item()
        return predicted, probs.cpu().numpy()
    



class ResNetMemeRegressor(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = self.load()

    def load(self, regressor_path=regressor_path):
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(torch.load(regressor_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    @BaseModel.predict_wrapper
    def predict(self, output):
        return output[0][0].item(), output[0][1].item(), output[0][2].item()



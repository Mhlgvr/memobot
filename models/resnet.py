import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from .inference import BaseModel


model_path = '/Users/mhlgvr/Documents/Yandex.Disk.localized/CU2/AI/Bootcamps 2025/memobot/data/resnet50.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResNetMemeClassifier(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = self.load(model_path)

    def load(self, model_path):
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 2)

        model.load_state_dict(torch.load(model_path, map_location=device))

        model.to(device)
        model.eval()
        return model
    
    def preprocess(self, img_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img = Image.open(img_path)
        img = transform(img)
        return img.to(device).unsqueeze(0)

    def predict(self, img):
        img = self.preprocess(img)
        with torch.no_grad():
            output = self.model(img)
        probs = output.softmax(dim=1)
        predicted = probs.argmax(dim=1).item()
        return predicted, probs.cpu().numpy()


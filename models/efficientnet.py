from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch

from .inference import BaseModel

model_path = '/Users/mhlgvr/Documents/Yandex.Disk.localized/CU2/AI/Bootcamps 2025/memobot/data/efficientnet.pth'

class EfficientNetMemeClassifier(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = self.load()


    def load(self):
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(1280, 2)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model.to(self.device)
        return model

    def preprocess(self, img_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        img = img.to(self.device).unsqueeze(0)
        return img
    
    def predict(self, img):
        img = self.preprocess(img)
        with torch.no_grad():
            output = self.model(img)
        probs = output.softmax(dim=1)
        predicted = probs.argmax(dim=1).item()
        return predicted, probs.cpu().numpy()

    

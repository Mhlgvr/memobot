import torch
from torchvision import transforms
from PIL import Image
from abc import ABC, abstractmethod
from functools import wraps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseModel(ABC):
    @abstractmethod
    def load(self):
        """
        Должен быть реализован в подклассе:
        должен возвращать модель с инициализированной архитектурой и 
        загруженными весами.
        """
        pass

    def preprocess(self, img_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        return img.to(device).unsqueeze(0)

    @staticmethod
    def predict_wrapper(func):
        @wraps(func)
        def wrapper(self, img_path):
            img = self.preprocess(img_path)
            with torch.no_grad():
                output = self.model(img)
                return func(self, output)
        return wrapper

    @abstractmethod
    def predict(self, img):
        pass


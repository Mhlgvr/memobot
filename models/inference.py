import torch
from torchvision import transforms
from PIL import Image
import abc


class BaseModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abc.abstractmethod
    def load(self, file_path): pass

    @abc.abstractmethod 
    def predict(self, img_path): pass

    @abc.abstractmethod
    def preprocess(self, img_path): pass

import torch
from torchvision import transforms
from PIL import Image
import abc
from resnet import init_model

model = init_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BaseModel(abc.ABC):
    @abc.abstractmethod              # оставим это на будущее
    def load(self, file_path): pass

    @abc.abstractmethod 
    def predict(self, img_path): pass

    @abc.abstractmethod
    def preprocess(self, img_path): pass




def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(img)

def predict(img_path):
    img = preprocess(img_path)
    with torch.no_grad():
        input = img.to(device).unsqueeze(0)
        logit = model(input)
    return logit.softmax(dim=1).argmax(dim=1)



if __name__ == "__main__":
    import sys
    img_path = sys.argv[1]
    result = predict(img_path)
    print("Predicted class:", result)



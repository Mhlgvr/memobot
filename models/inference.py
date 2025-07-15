import torch
from models.resnet import init_model
from torchvision import transforms
from PIL import Image
from typing import Union

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_model()

def predict(img_path):
    img = preprocess(img_path)
    with torch.no_grad():
        input = img.to(device).unsqueeze(0)
        logit = model(input)
    # return logit.softmax(dim=1).argmax(dim=1)
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
        print(param.data)
    return input


def preprocess(img_path: Union[str, bytes]) -> torch.Tensor:
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(img)


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1]
    result = predict(img_path)
    print("Predicted class:", result)



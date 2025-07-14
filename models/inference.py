import torch
from resnet import model
from torchvision import transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(img):
    input_tensor = preprocess(img)
    with torch.no_grad():
        logit = model(input_tensor)
        prob = logit.softmax(dim=1)
        pred = prob.argmax(dim=1)
    return pred


def preprocess(img):
    img = Image.open(img)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = transform(img)
    return tensor.to(device).unsqueeze(0)


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1]
    result = predict(img_path)
    print("Predicted class:", result)



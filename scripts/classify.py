# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.resnet import ResNetMemeClassifier
from models.efficientnet import EfficientNetMemeClassifier

class_names = ['not meme', 'meme']

def load(model):
    if model.lower() == 'resnet':
        return ResNetMemeClassifier()
    elif model.lower() == 'efficientnet':
        return EfficientNetMemeClassifier()
    else:
        raise NotImplementedError(f'Unknown model: {model}')

if __name__ == '__main__':
    import sys
    model = sys.argv[1]
    img = sys.argv[2]
    model = load(model)
    result = model.predict(img)
    print(f'Predicted class: {class_names[result[0]]}. Probs: {result[1]}')

